import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import gc
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from accelerate import Accelerator

# =============== IMPORT FUNCIONES ======================
from models.model_1graph import weights_init,get_keep_ia3_model
from utils.data_load import create_file_label_df, RAM_WSI_Dataset, collate_fn
from utils.utils import compute_metrics,run_epoch


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
gc.collect()

# =================== HIPERPARÉMETROS ===================
LEARNING_RATES = [1e-3]#, 1e-5, 1e-6,1e-4]
EPOCHS_RANGE = [100]
BATCH_SIZES = [16]#, 128, 64,32]
PATIENCE = 20
RANDOM_SEED = 98
num_workers = 4

# =================== CONFIGURACION MODELOS ===================
BASE_PATH = "/workspace/results"

use_amp = torch.cuda.is_available()  # mixed precision if GPU available

for lr, batch, epochs in itertools.product(LEARNING_RATES, BATCH_SIZES, EPOCHS_RANGE):
    name = f"try_{lr}_{batch}_{epochs}"
    
    params = {
        "batch": batch,
        "lr": lr,
        "epochs": epochs
    }

    # Multi-GPU accelerator init
    accelerator = Accelerator()
    device = accelerator.device

    print(f"Usando dispositivo: {device}", flush = True)
    
    paths = {
        'output_file': os.path.join(BASE_PATH, name, 'output.txt'),
        'plot_kfold_path': os.path.join(BASE_PATH, name, 'k_fold_plot.png'),
        'results_sicap': os.path.join(BASE_PATH, name),
        'results_path': os.path.join(BASE_PATH, name, 'result_mdl.xlsx'),
        'model_path': os.path.join(BASE_PATH, name, 'best_model.pth'),
        'embedding_path':  '/workspace/data/pandas/patches',
        'data_path':  '/workspace/data/datos_PANDAS.xlsx',
        'sicap_path_embeddings': '/workspace/data/features_SICAP/patches',
        'excel_path_sicap': '/workspace/data/wsi_labels_changed.xlsx',
    }

    # Model Configuration 
    model = get_keep_ia3_model().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Loading Data
    print("Preparing Data ...")
    df_train = create_file_label_df(paths["embedding_path"], paths["data_path"])

    label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
    df_train['gleason_primary_mapped'] = df_train['gleason_primary'].map(label_mapping)
    df_train['gleason_secondary_mapped'] = df_train['gleason_secondary'].map(label_mapping)
    df_train['gleason_score'] = df_train['gleason_score'].astype(int)

    # spliting data
    train_val_idx, test_idx = train_test_split(df_train.index, test_size=0.15, stratify=df_train["gleason_score"])

    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.10, stratify=df_train.loc[train_val_idx]["gleason_score"])
   
    # DataLoaders
    dl_kwargs = {"batch_size": params["batch"], "collate_fn": collate_fn, "num_workers": num_workers}
    if torch.cuda.is_available():
        dl_kwargs["pin_memory"] = True

    print("Loading embeddings for Training ...")
    train_loader = DataLoader(RAM_WSI_Dataset(df_train, train_idx,random_subsets=True ), shuffle=True, **dl_kwargs)
    print("Loading embeddings for Validation ...")
    val_loader = DataLoader(RAM_WSI_Dataset(df_train, val_idx), shuffle=False, **dl_kwargs)
    print("Loading embeddings for Test ...")
    test_loader = DataLoader(RAM_WSI_Dataset(df_train, test_idx), shuffle=False, **dl_kwargs)

    # loss / optimizer / scheduler
    loss = nn.BCEWithLogitsLoss()
    model.apply(weights_init)
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=PATIENCE, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # si no hay GPU, no hace nada

    torch.cuda.empty_cache()
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader )
    
    best_loss = float("inf")
    no_improve = 0
    best_validation_loss=1000

    for epoch in range(params["epochs"]):
        print(f"\nEpoch {epoch}")
        # TRAIN
        model.train()
        train_metrics, train_loss = run_epoch(model, train_loader, device, accelerator, loss, optimizer, mode='train')

        # VALIDATION
        model.eval()
        val_metrics, val_loss = run_epoch(model, val_loader, device, accelerator, loss, optimizer)
       
        scheduler.step(val_loss)
       
        #  Early Stopping / Checkpoint
        if val_loss < best_validation_loss:
            best_validation_loss, no_improve = val_loss, 0
            if accelerator.is_main_process:
                accelerator.save_model(model, paths['model_path'])
                print(f"Mejor modelo guardado en {paths['model_path']} con val_loss: {best_validation_loss:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print('Early stopping activado - Entrenamiento finalizado ', flush=True)
                break

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_metrics[0]:.4f}")
        print(f"           Val Loss: {val_loss:.4f} Acc: {val_metrics[0]:.4f} Kappa: {val_metrics[-1]:.4f}")

        with open(paths['output_file'], "a") as f:
            f.write(
                f"Epoch {epoch}|    Train Loss: {train_loss:.4f} Acc: {train_metrics[0]:.4f}\n"
                f"          Val Loss: {val_loss:.4f}         Acc: {val_metrics[0]:.4f}    Kappa: {val_metrics[-1]:.4f}\n")

    # TEST
    model.eval()
    test_metrics, test_loss = run_epoch(model, val_loader, device, accelerator, loss, optimizer)   

    print(f"Test Loss: {test_loss:.4f} Acc: {test_metrics[0]:.4f} Kappa: {test_metrics[-1]:.4f}")
    with open(paths['output_file'], "a") as f:
        f.write(f" Test Loss: {test_loss:.4f}       Acc: {test_metrics[0]:.4f}   Kappa: {test_metrics[-1]:.4f}\n")
        f.write(f"Acuracy: {test_metrics[0]}, | Precision: {test_metrics[1]} | Recall : {test_metrics[2]}| F1 Score: {test_metrics[3]}")