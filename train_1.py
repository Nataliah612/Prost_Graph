import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader
import gc
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============== IMPORT FUNCIONES ======================
from models.model_1graph import get_keep_ia3_model, get_keep_lora_model,model_graph, device, weights_init
from models.restrictions import loss_restrictions
from utils.data_load import create_file_label_df, RAM_WSI_Dataset, collate_fn
from utils.utils import compute_metrics


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
gc.collect()

# =================== CONFIGURACIÓN ===================
BASE_PATH = "/workspace/Prost_Graph/results"
MODEL_TYPE = 3
LEARNING_RATES = [1e-2]#, 1e-5, 1e-6,1e-4]
EPOCHS_RANGE = [100]
BATCH_SIZES = [128]#, 128, 64,32]
PATIENCE = 20
RANDOM_SEED = 98

# =================== Accelerate ===================

print(f"✅ Usando dispositivo: {device}")
#torch.to.empty_cache()



# =================== FUNCIONES AUXILIARES ===================
def load_model(type_id, name):
    model_map = {3: (get_keep_ia3_model, name)}
    model_fn, path = model_map.get(type_id, (get_keep_ia3_model, 'other'))
    model = model_fn()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    os.makedirs(os.path.join(BASE_PATH, path), exist_ok= True)
    paths = {
        'output_file': os.path.join(BASE_PATH, path, 'output.txt'),
        'plot_kfold_path': os.path.join(BASE_PATH, path, 'k_fold_plot.png'),
        'results_sicap': os.path.join(BASE_PATH, path),
        'results_path': os.path.join(BASE_PATH, path, 'result_mdl.xlsx'),
        'model_path': os.path.join(BASE_PATH, path, 'best_model.pth'),
        'embedding_path':  '/workspace/data/pandas/patches',
        'data_path':  '/workspace/data/datos_PANDAS.xlsx',
        'sicap_path_embeddings': '/workspace/data/features_SICAP/patches',
        'excel_path_sicap': '/workspace/data/wsi_labels_changed.xlsx',
    }

    return model, paths

def train_and_evaluate(model, model_id, model_path, params, paths):
    
    print("Preparing Data ...")

    df_train = create_file_label_df(paths["embedding_path"], paths["data_path"])
    label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}

    df_train['gleason_primary_mapped'] = df_train['gleason_primary'].map(label_mapping)
    df_train['gleason_secondary_mapped'] = df_train['gleason_secondary'].map(label_mapping)

    train_val_idx, test_idx = train_test_split(
        df_train.index, test_size=0.15,
        stratify=df_train["gleason_score"]
    )

    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.10,
        stratify=df_train.loc[train_val_idx]["gleason_score"]
    )

    print("Loading embeddings for Training ...")
    data_train = RAM_WSI_Dataset(df_train, train_idx, random_subsets=True, len_subset=int(0.3 * len(train_idx)))

    train_loader = DataLoader(data_train, batch_size=params["batch"], shuffle=True,  collate_fn=collate_fn)
    print("Loading embeddings for Validation ...")
    val_loader   = DataLoader(RAM_WSI_Dataset(df_train, val_idx), batch_size=params["batch"], shuffle=False, collate_fn=collate_fn)
    print("Loading embeddings for Test ...")
    test_loader = DataLoader(RAM_WSI_Dataset(df_train, test_idx), batch_size=params["batch"],shuffle=False, collate_fn=collate_fn)

    # ------------ perdidas y optimizadores ------------------------
  
    loss_sigm = nn.BCEWithLogitsLoss()

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.apply(weights_init)
   
    optimizer = optim.SGD(model.parameters(), lr=params["lr"], weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",  # porque usamos val_loss (minimizar)
        factor=0.1,  # reduce LR a un 10% cuando no mejora
        patience=PATIENCE,  # cuántas épocas esperar sin mejora
        #verbose=True  # imprime cambios de LR
    )

    best_loss, no_improve = float("inf"), 0

    for epoch in range(params["epochs"]):
        print(f"\nEpoch {epoch}")

        # TRAINING
        model.train()
        train_loss_sum = 0.0
        train_correct  = torch.zeros(1, dtype=torch.long, device=device)
        train_seen     = torch.zeros(1, dtype=torch.long, device=device)

        for inputs, y in tqdm(train_loader, desc="Training"):
            
            y = y.to(device)

            y_onehot = torch.zeros(y.shape[0], 4).to(device)
            y_onehot.scatter_(1, y, 1)  

            out1 = model(inputs.to(device))
            loss = loss_sigm(out1,y_onehot.to(device)) #+ loss_restrictions(out1, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_sz = inputs.size(0)
            train_loss_sum += loss.item() * batch_sz

            outputs = torch.sigmoid(out1)  # si usas BCEWithLogitsLoss
            top2 = torch.topk(outputs, 2, dim=1)  
            preds_ok = ((top2.indices == y).all(dim=1)).sum()
            
            train_correct += preds_ok
            train_seen    += batch_sz

        train_loss    = train_loss_sum / train_seen
        train_acc     = train_correct / train_seen

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_correct  = torch.zeros(1, dtype=torch.long, device=device)
        val_seen     = torch.zeros(1, dtype=torch.long, device=device)
        all_y1, all_p1,  = [], []

        with torch.no_grad():
            for inputs, y in tqdm(val_loader, desc="Validation"):
                y = y.to(device)
                #y = torch.tensor([[label_mapping[int(v.item())] for v in row] for row in y],device=device)

                y_onehot = torch.zeros(y.shape[0], 4).to(device)
                y_onehot.scatter_(1, y, 1)  

                out1 = model(inputs.to(device))
                loss = loss_sigm(out1,y_onehot.to(device))#+ loss_restrictions(out1, device=device)

                batch_sz = inputs.size(0)
                val_loss_sum += loss.item() * batch_sz

                outputs = torch.sigmoid(out1)  # si usas BCEWithLogitsLoss
                top2 = torch.topk(outputs, 2, dim=1)  # los 2 valores más altos por muestra

                all_y1.append(y.sum(dim=1)); all_p1.append(top2.indices.sum(dim=1))
                
                preds_ok = ((top2.indices == y).all(dim=1)).sum()
                val_correct += preds_ok
                val_seen    += batch_sz

        val_loss    = val_loss_sum / val_seen
        val_acc     = val_correct / val_seen

        g_ty_sum = torch.cat(all_y1)[:val_seen]
        g_tp_sum = torch.cat(all_p1)[:val_seen]

        val_kappa = compute_metrics(g_ty_sum, g_tp_sum)[-1] 

        scheduler.step(val_loss)

        # EARLY-STOP / CHECKPOINT
        if val_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss, no_improve = val_loss, 0
           
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

        print(f"Epoch {epoch} | Train Loss: {train_loss[0]:.4f} Acc: {train_acc[0]:.4f}")
        print(f"           Val Loss: {val_loss[0]:.4f} Acc: {val_acc[0]:.4f} Kappa: {val_kappa:.4f}")
        

        with open(paths['output_file'], "a") as f:
            f.write(f"Epoch {epoch}|    Train Loss: {train_loss[0]:.4f} Acc: {train_acc[0]:.4f}\n"
                    f"          Val Loss: {val_loss[0]:.4f}         Acc: {val_acc[0]:.4f}    Kappa: {val_kappa:.4f}\n")
    # TEST
    model.eval()
    test_loss_sum = 0.0
    test_correct  = torch.zeros(1, dtype=torch.long, device=device)
    test_seen     = torch.zeros(1, dtype=torch.long, device=device)
  
    all_y1, all_p1 = [],[]
    with torch.no_grad():
        for inputs, y in tqdm(test_loader, desc="Test"):
            #y = torch.tensor([[label_mapping[int(v.item())] for v in row] for row in y],device=device)
            y = y.to(device)

            y_onehot = torch.zeros(y.shape[0], 4).to(device)
            y_onehot.scatter_(1, y.to(device), 1)  

            out1 = model(inputs.to(device))
            loss = loss_sigm(out1,y_onehot.to(device))#+ loss_restrictions(out1, device=device)

            batch_sz = inputs.size(0)
            test_loss_sum += loss.item() * batch_sz
            
            outputs = torch.sigmoid(out1)  # si usas BCEWithLogitsLoss
            top2 = torch.topk(outputs, 2, dim=1)  # los 2 valores más altos por muestra

            all_y1.append(y.sum(dim=1)); all_p1.append(top2.indices.sum(dim=1))
            
            preds_ok = ((top2.indices == y).all(dim=1)).sum()
            test_correct += preds_ok
            test_seen    += batch_sz


    test_loss    = test_loss_sum / test_seen
    test_acc     = test_correct / test_seen

    g_ty_sum = torch.cat(all_y1)[:test_seen]
    g_tp_sum = torch.cat(all_p1)[:test_seen]

    test_kappa = compute_metrics(g_ty_sum, g_tp_sum)[-1] 
    metrics = compute_metrics(g_ty_sum, g_tp_sum)
         
    print(f"Test Loss: {test_loss[0]:.4f} Acc: {test_acc[0]:.4f} Kappa: {test_kappa:.4f}")
    with open(paths['output_file'], "a") as f:
        f.write(  f" Test Loss: {test_loss[0]:.4f}       Acc: {test_acc[0]:.4f}   Kappa: {test_kappa:.4f}\n")
        f.write(  f"Acuracy: {metrics[0]}, | Precision: {metrics[1]} | Recall : {metrics[2]}| F1 Score: {metrics[3]}")

# =============== EJECUCIÓN ===================
if __name__ == "__main__":
    for lr, batch, epochs in itertools.product(LEARNING_RATES, BATCH_SIZES, EPOCHS_RANGE):
        name = f"try_{lr}_{batch}_{epochs}"
        model, paths = load_model(MODEL_TYPE, name)

        with open(paths['output_file'], "w") as f:
                    f.write(f"Modelo: {name} \n")
        params = {
            "batch": batch,
            "lr": lr,
            "epochs": epochs
        }

        trained_model = train_and_evaluate(model, MODEL_TYPE, paths["model_path"], params, paths)