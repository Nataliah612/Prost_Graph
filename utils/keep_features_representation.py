import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import argparse
import random 
import os 
import numpy as np 

from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model,IA3Config
from accelerate import Accelerator

from data_load import create_file_label_df, RAM_WSI_Dataset

use_amp = torch.cuda.is_available()  # mixed precision if GPU available

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL + Patch Classifier")

    parser.add_argument("--data_panda_patches", type=str, default='/workspace/data/pandas/patches', help="Ruta de carpeta con imágenes WSI de Panda en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_panda_path", type=str, default='/workspace/data/datos_PANDAS.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de Panda asociadas a la ruta del .h5")
    parser.add_argument("--data_sicap_patches", type=str, default='/workspace/data/features_SICAP/patches', help="Ruta de carpeta con imágenes WSI de SICAP en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_sicap_path", type=str, default='/workspace/data/wsi_labels_changed_SICAP.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")
    parser.add_argument("--model_path", type=str, default='/workspace/results/try_0.1_4_20/best_model.pth/model.safetensors', help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")
    parser.add_argument("--results", type=str, default='/workspace/results', help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")

    return parser.parse_args()
def set_multiple_seed(seed: int = 42, deterministic: bool = True):
    """
    Fija las semillas para reproducibilidad en Python, NumPy y PyTorch.
    
    Args:
        seed (int): Valor de la semilla a usar (por defecto 42).
        deterministic (bool): Si True, fuerza operaciones deterministas en CUDA 
                              (puede afectar la velocidad).
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU y GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para multi-GPU
    
    if deterministic:
        # Hace que las operaciones de PyTorch sean deterministas
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Permite a cuDNN seleccionar algoritmos más rápidos pero no reproducibles
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"[INFO] Semilla fijada en {seed} (deterministic={deterministic})")

# =================== GENERAL CONFIGURACION ===================

use_amp = torch.cuda.is_available()  # mixed precision if GPU available
args = parse_args()


# Multi-GPU accelerator init
accelerator = Accelerator()
device = accelerator.device
print(f"Usando dispositivo: {device}", flush = True)
torch.cuda.empty_cache()

set_multiple_seed(98)

def get_features(model, dataloader):
    feats = []
    with torch.no_grad():
        for x, _ in dataloader:
            feats.append(model(x).cpu())
    return torch.cat(feats).numpy()


def compute_class_weights(labels):
    labels = torch.tensor(labels)
    num_classes = 4
    class_counts = torch.bincount(labels, minlength=num_classes)

    weights = 1.0 / (class_counts.float() + 1e-8)
    weights = weights / weights.sum() * num_classes
    
    return weights.to(dtype=torch.float, device=device)

def get_keep_ia3_model():
    model_name = "Astaxanthin/KEEP" # Modelo preentrenado 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Modelo cuantificado en 4 bits 
        bnb_4bit_compute_dtype=torch.float16,   # Float16 para maentener la velocidad y precision 
        bnb_4bit_use_double_quant=True,         # Usa doble cuanticicacion
        bnb_4bit_quant_type="nf4",              # Variante de tipo de cuqntificacion de 4 bits 
    )
   
    model = AutoModel.from_pretrained( # Carga el modelo 
        model_name,
        trust_remote_code=True,             # Ejecuta el codio en remoto
        quantization_config=bnb_config,     # Introduce la configuracion de carga del modelo
        force_download=True,                # Descarga del modelo 
        low_cpu_mem_usage = True            # Reduce memoria en la carga
    )
        
    # Activar gradient checkpointing para reducir VRAM
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    
    model.config.use_cache = False  # Reducir uso de memoria durante forward

    # Identificar módulos lineales donde aplicar IA3
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]#, "mlp.fc2"]
        ):
            target_modules.append(name)

    ia3_config = IA3Config(
        perf_type='IA3',
        target_modules=target_modules,
        feedforward_modules=target_modules,
        task_type="FEATURE_EXTRACTION" #SEQ_CLS
    )

    ia3_model = get_peft_model(model, ia3_config)
    ia3_model.print_trainable_parameters()
   
    model = torch.compile(model) # Activar torch.compile si usas PyTorch 2.0+
    # Congelar TODO menos IA3
    for name, param in ia3_model.named_parameters():
        #if "fc1.ia3" not in name :
        if "blocks.23.mlp.fc1.ia3" not in name and "blocks.22.mlp.fc1.ia3" not in name and "blocks.21.mlp.fc1.ia3" not in name :
            param.requires_grad = False
    
    return ia3_model

# ------------ Data --------------------

df_test = create_file_label_df(args.data_sicap_patches, args.data_sicap_path, NAME = 'SICAP')

label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
df_test['gleason_primary_mapped'] = df_test['gleason_primary'].map(label_mapping)
df_test['gleason_secondary_mapped'] = df_test['gleason_secondary'].map(label_mapping)
df_test['gleason_score'] = df_test['gleason_score'].astype(int)

print("Loading embeddings for Inference ...")
test_loader = DataLoader(RAM_WSI_Dataset(df_test, df_test.index), shuffle=False)

# ---------- 1) Modelo congelado ----------
model_frozen = get_keep_ia3_model()
for p in model_frozen.parameters():
    p.requires_grad = False

features_frozen = get_features(model_frozen, test_loader)

# ---------- 2) Modelo entrenado ----------
model_trained = get_keep_ia3_model()
model_trained.load_state_dict(torch.load(args.model_path))

layers_to_extract = ['fc1', 'fc2']
weights_subset = {k: v for k, v in model_trained.state_dict().items() if any(l in k for l in layers_to_extract)}

new_model = get_peft_model()
new_state = new_model.state_dict()
new_state.update(weights_subset)  # solo reemplaza las capas que coincidan
new_model.load_state_dict(new_state)

features_trained = get_features(new_model, test_loader)

# ---------- t-SNE ----------
tsne = TSNE(n_components=2, perplexity=30)

t1 = tsne.fit_transform(features_frozen)
t2 = tsne.fit_transform(features_trained)

# ---------- Plot ----------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(t1[:,0], t1[:,1])
plt.title("Modelo congelado")

plt.subplot(1,2,2)
plt.scatter(t2[:,0], t2[:,1])
plt.title("Modelo entrenado")

plt.savefig(args.results)
