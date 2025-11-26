import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import os
import itertools
import gc
import time 
import argparse

from accelerate import Accelerator

# =============== IMPORT FUNCIONES ======================
from models.model_1graph import get_keep_ia3_model
from utils.data_load import create_file_label_df, RAM_WSI_Dataset, collate_fn
from utils.utils import run_epoch, compute_class_weights, set_multiple_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL + Patch Classifier")

    parser.add_argument("--data_panda_patches", type=str, default='/workspace/data/pandas/patches', help="Ruta de carpeta con imágenes WSI de Panda en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_panda_path", type=str, default='/workspace/data/datos_PANDAS.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de Panda asociadas a la ruta del .h5")
    parser.add_argument("--data_sicap_patches", type=str, default='/workspace/data/features_SICAP/patches', help="Ruta de carpeta con imágenes WSI de SICAP en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_sicap_path", type=str, default='/workspace/data/wsi_labels_changed_SICAP.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")

    parser.add_argument("--results_path", type=str, default='/workspace/results', help="Ruta donde se guardaran los resultados")

    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño de batch")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--patience", type=int, default=20, help="Paciencia para early stopping")

    parser.add_argument("--seed", type=int, default=98, help="Semilla aleatoria")
    parser.add_argument("--num_workers", type=int, default=2, help="Numero de paralelizaciones")

    parser.add_argument("--alpha_pe", default=0.1, type=float)
    parser.add_argument("--alpha_pc", default=0.1, type=float)

    return parser.parse_args()

# =================== GENERAL CONFIGURACION ===================

use_amp = torch.cuda.is_available()  # mixed precision if GPU available
args = parse_args()

# Multi-GPU accelerator init
accelerator = Accelerator()
device = accelerator.device
print(f"Usando dispositivo: {device}", flush = True)
torch.cuda.empty_cache()

set_multiple_seed(args.seed)

# =================== PATHS DEFINITION ===================
name = f"try_{args.learning_rate}_{args.batch_size}_{args.epochs}"

results_path = os.path.join(args.results_path, name)
os.makedirs(results_path, exist_ok=True)

paths = {
    'output_file': os.path.join(results_path, 'output.txt'),
    'plot_kfold_path': os.path.join(results_path, 'k_fold_plot.png'),
    'results_sicap': os.path.join(results_path),
    'results_path': os.path.join(results_path, 'result_mdl.xlsx'),
    'model_path': os.path.join(results_path, 'best_model.pth'),
}

# =================== PMODEL CONFIGURATION ===================
model = get_keep_ia3_model().to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# =================== LOADING DATA ===================
print("Preparing DataSet SICAP ...")

with open(paths['output_file'], "w") as f:
        f.write(f"Creating Dataset SICAP \n")
df_train = create_file_label_df(args.data_sicap_patches, args.data_sicap_path, NAME = 'SICAP')

label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
df_train['gleason_primary_mapped'] = df_train['gleason_primary'].map(label_mapping)
df_train['gleason_secondary_mapped'] = df_train['gleason_secondary'].map(label_mapping)
df_train['gleason_score'] = df_train['gleason_score'].astype(int)

# =================== DATA SPLIT ===================
train_val_idx, test_idx = train_test_split(df_train.index, test_size=0.15, stratify=df_train["gleason_score"])
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.10, stratify=df_train.loc[train_val_idx]["gleason_score"])

# =================== DATALOADERS ===================
dl_kwargs = {"batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.num_workers}
if torch.cuda.is_available():
    dl_kwargs["pin_memory"] = True

inicio__load = time.time() 
print("Loading embeddings for Training ...")
train_loader = DataLoader(RAM_WSI_Dataset(df_train, train_idx), shuffle=True, **dl_kwargs)
print("Loading embeddings for Validation ...")
val_loader = DataLoader(RAM_WSI_Dataset(df_train, val_idx), shuffle=False, **dl_kwargs)
print("Loading embeddings for Test ...")
test_loader = DataLoader(RAM_WSI_Dataset(df_train, test_idx), shuffle=False, **dl_kwargs)
fin__load = time.time()

time_load = fin__load - inicio__load

with open(paths['output_file'], "a") as f:
    f.write(f"Time for loading data : {time_load} \n")

# =================== WEIGHTS CALCUALTION ===================
primary_weights = compute_class_weights(df_train['gleason_primary_mapped'])
secondary_weights = compute_class_weights(df_train['gleason_secondary_mapped'])
sum_weights = primary_weights + secondary_weights

# =================== TRAINING PARAMETHERS ===================
loss = nn.BCEWithLogitsLoss(pos_weight=sum_weights) 

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum) # Adam -> SGD
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler )

best_loss = float("inf")
no_improve = 0

# =================== TRAINING ===================
for epoch in range(args.epochs):
    print(f"\nEpoch {epoch}")
    with open(paths['output_file'], "a") as f:
        f.write(f"\nEpoch {epoch}\n")
    
    # TRAIN
    model.train()
    train_metrics, train_loss = run_epoch(model, train_loader, device, accelerator, loss, optimizer, mode='train', args=args)

    # VALIDATION
    model.eval()
    val_metrics, val_loss = run_epoch(model, val_loader, device, accelerator, loss, optimizer, mode = 'val',args=args)
    
    scheduler.step(val_loss)
    
    #  Early Stopping / Checkpoint
    accelerator.wait_for_everyone()
    if val_loss < best_loss:
        best_loss, no_improve = val_loss, 0
        if accelerator.is_main_process:
            accelerator.save_model(model, paths['model_path'])
            print(f"Mejor modelo guardado en {paths['model_path']} con val_loss: {best_loss:.4f}", flush=True)
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print('Early stopping activado - Entrenamiento finalizado ', flush=True)
            break

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_metrics['global'][0]:.4f}")
    print(f"           Val Loss: {val_loss:.4f} Acc_global: {val_metrics['global'][0]:.4f} Kappa_global: {val_metrics['global'][-1]:.4f} | Acc_primary: {val_metrics['primary'][0]:.4f} Kappa_primary: {val_metrics['primary'][-1]:.4f} |Acc_secondary: {val_metrics['secondary'][0]:.4f} Kappa_secondary: {val_metrics['secondary'][-1]:.4f} ")

    with open(paths['output_file'], "a") as f: 
        f.write(f"Epoch {epoch}|    Train Loss: {train_loss:.4f} Acc: {train_metrics['global'][0]:.4f}\n")
        f.write(f"          Val Loss: {val_loss:.4f} Acc_global: {val_metrics['global'][0]:.4f} Kappa_global: {val_metrics['global'][-1]:.4f} | Acc_primary: {val_metrics['primary'][0]:.4f} Kappa_primary: {val_metrics['primary'][-1]:.4f} |Acc_secondary: {val_metrics['secondary'][0]:.4f} Kappa_secondary: {val_metrics['secondary'][-1]:.4f} ")
        
# TEST
model.eval()
test_metrics, test_loss = run_epoch(model, test_loader, device, accelerator, loss, optimizer, mode='test',args=args)   

print(f"Test Loss: {test_loss:.4f} Acc: {test_metrics[0]:.4f} Kappa: {test_metrics[-1]:.4f}")
print(f"Test Loss: {test_loss:.4f} Acc_global: {test_metrics['global'][0]:.4f} Kappa_global: {test_metrics['global'][-1]:.4f} | Acc_primary: {test_metrics['primary'][0]:.4f} Kappa_primary: {test_metrics['primary'][-1]:.4f} |Acc_secondary: {test_metrics['secondary'][0]:.4f} Kappa_secondary: {test_metrics['secondary'][-1]:.4f} ")
with open(paths['output_file'], "a") as f:
    f.write(f" Test Loss: {test_loss:.4f}  Acc_global: {test_metrics['global'][0]:.4f} Kappa_global: {test_metrics['global'][-1]:.4f} | Acc_primary: {test_metrics['primary'][0]:.4f} Kappa_primary: {test_metrics['primary'][-1]:.4f} |Acc_secondary: {test_metrics['secondary'][0]:.4f} Kappa_secondary: {test_metrics['secondary'][-1]:.4f}\n")
    f.write(f"Acuracy: {test_metrics[0]}, | Precision: {test_metrics[1]} | Recall : {test_metrics[2]}| F1 Score: {test_metrics[3]}")
    f.write(f"Métricas : {test_metrics}")