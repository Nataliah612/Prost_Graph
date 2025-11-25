import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import softmax

import os
from transformers import AutoModel, BitsAndBytesConfig

from peft import get_peft_model,IA3Config #, LoraConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available(), "  n_gpus:", torch.cuda.device_count())

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
        target_modules=target_modules,
        feedforward_modules=target_modules,
        task_type="FEATURE_EXTRACTION"
    )

    ia3_model = get_peft_model(model, ia3_config)
   
    model = torch.compile(model) # Activar torch.compile si usas PyTorch 2.0+
    # Congelar TODO menos IA3
    for name, param in ia3_model.named_parameters():
        #if "fc1.ia3" not in name :
        if "blocks.23.mlp.fc1.ia3" not in name and "blocks.22.mlp.fc1.ia3" not in name and "blocks.21.mlp.fc1.ia3" not in name :
            param.requires_grad = False
    # 7. Envolver con tu modelo final
    net = KEEPForGleason(ia3_model, model_graph)
    return net

class KEEPForGleason(nn.Module):
    def __init__(self, base_model, classifier_model):
        super().__init__()
        self.base = base_model
        self.classifier = classifier_model()
        self.classifier.apply(weights_init)

    def forward(self, images):
        # Codificación del encoder visual
        features_list = [torch.stack([self.base.encode_image(patch.unsqueeze(dim=0).to(dtype=torch.float16)) for patch in img.squeeze(0)])for img in images]
        logits = self.classifier(features_list)
        return logits

class model_graph(nn.Module):
    def __init__(self):
        super(model_graph, self).__init__()
        self.graphgat = gat()
        self.to(device)

    def forward(self, x):
        """
        x: tensor (B, N, D) 
        """
        graphs_list = [grafo_fully_connected(xi.squeeze(dim=1)) for xi in x]
       
        out, w1 = self.graphgat(graphs_list)
        return out, w1

def grafo_fully_connected(x):
    num_nodos = x.shape[0]
    
    # Creamos todas las combinaciones posibles
    fila = torch.arange(num_nodos).repeat_interleave(num_nodos)
    col = torch.arange(num_nodos).repeat(num_nodos)
    
    # Quitamos autoconexiones si quieres
    mask = fila != col
    edge_index = torch.stack([fila[mask], col[mask]], dim=0)

    return Data(x=x, edge_index=edge_index)

class gat(torch.nn.Module):
    def __init__(self, heads=8):
        super(gat, self).__init__()
        self.gat1 = GATv2Conv(768, 256, heads=heads, concat=False)
        self.gat2 = GATv2Conv(256, 64, heads=heads, concat=False)
         # Debe aceptar batch
        self.Aggr = Aggregate_Attnt(L=256, I=1)
        #self.Aggr.apply(weights_init)
        self.fc = mlp(in_dim=64, out_dim=4)

    def forward(self, graph_list):
        
        # 1) Empaquetar graphs en un solo batch
        batch = Batch.from_data_list(graph_list)
        batch = batch.to(device)

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        lengths = batch.ptr[1:] - batch.ptr[:-1]   # patches por WSI

        # 2) GATConv 
        z1 = self.gat1(x, edge_index)
        z1 = F.relu(z1)

        z2 = self.gat2(z1, edge_index)
        z2 = F.relu(z2)

        # 3) Clasificador por patch
        logits = self.fc(z2)
        probs = torch.softmax(logits, dim=1)

        # 4) MIL aggregation por WSI usando batch_idx   
        w2 = self.Aggr(z1, batch_idx)
        
        # 5) Weighted max per WSI
        weighted = w2 * probs             # (N_total, 4)

        # 6) Reconstruir patch predictions por WSI
        patch_output = torch.split(probs, lengths.tolist(), dim=0)
        out_wsi = [torch.sum(weighted,dim=0) for p in patch_output]
        #out_wsi = [torch.max(weighted,dim=0)[0] for p in patch_output]

        return torch.stack(out_wsi), patch_output

class Aggregate_Attnt(nn.Module):
    def __init__(self, L=512, I=1):
        super().__init__()
        D = L // 2
        self.branch_1 = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.branch_2 = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.branch_3 = nn.Linear(D, I)

    def forward(self, x: Tensor, batch_idx: Tensor):
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa.mul(xb)
        A = self.branch_3(A)
        gate = softmax(A, index=batch_idx, num_nodes=x.size(0))
        return gate

class mlp(nn.Module):
    def __init__(self, in_dim=64, out_dim=4):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

'''from transformers import AutoModel, BitsAndBytesConfig
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import torch

# ----------------------------------------------------
# Inicializar KEEP + LoRA con 4-bit Quantization
# ----------------------------------------------------
def get_keep_lora_model():
    model_name = "Astaxanthin/KEEP"
    
    # 1. Configurar la Cuantización de 4 bits (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # <--- Carga el modelo en 4-bits
        bnb_4bit_compute_dtype=torch.bfloat16, # <--- Usa bfloat16 para el cómputo (mejor que float16 si tu GPU lo soporta)
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        quantization_config=bnb_config, # <--- Aplicar la configuración
        low_cpu_mem_usage=True
    )
    
    # Asegúrate de que el modelo base esté en modo train para LoRA
    # aunque la mayoría de los parámetros están congelados.
    

    # ... (El resto de tu código LoRA/PEFT sigue aquí)
    
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]):
            target_modules.append(name)

    lora_config = LoraConfig(
        r=4,  # más capacidad representativa
        lora_alpha=8,  # proporcional al rango
        target_modules=target_modules,
        lora_dropout=0.3,  # regulariza para dataset pequeño/desbalanceado
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    lora_model = get_peft_model(model, lora_config)
    
    # 3. Optimización: Desactivar cache y activar Gradient Checkpointing
    lora_model.config.use_cache = False # Desactiva la cache para reducir memoria en la inferencia
    if hasattr(lora_model, "gradient_checkpointing"):
        lora_model.gradient_checkpointing = True # <--- MUY IMPORTANTE
    
    # 5. Compilación (PyTorch 2.0+)
    lora_model = torch.compile(lora_model)

    # 6. Envolver en tu modelo final
    net = KEEPForGleason(lora_model, model_graph)

    return net'''
