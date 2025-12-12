import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import os
import gc
import csv
import time 
import argparse

from accelerate import Accelerator

# =============== IMPORT FUNCIONES ======================
from models.keep_model import get_keep_ia3_model
from models.CNN import CNNEmbedder

from utils.data_load import create_file_label_df, WSI_Dataset, collate_fn
from utils.utils_train import run_epoch, compute_class_weights, set_multiple_seed
from utils.utils_metrics import log_epoch_metrics, save_metrics_to_csv

os.environ["ACCELERATE_DISABLE_TRANSFORMER_ENGINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL + Patch Classifier")

    parser.add_argument("--data_panda_patches", type=str, default='/workspace/data/pandas/patches', help="Ruta de carpeta con imágenes WSI de Panda en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_panda_path", type=str, default='/workspace/data/datos_PANDAS.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de Panda asociadas a la ruta del .h5")
    parser.add_argument("--data_sicap_patches", type=str, default='/workspace/data/features_SICAP/patches', help="Ruta de carpeta con imágenes WSI de SICAP en .h5 (Dentro deben contener parches)")
    parser.add_argument("--data_sicap_path", type=str, default='/workspace/data/wsi_labels_changed_SICAP.xlsx', help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")

    parser.add_argument("--inference",action="store_true",default=False, help="Usar una BBDD para hacer inferencia despues del entrenamiento.")
    parser.add_argument("--small_dataset",action="store_true",default=False, help="Usar un subset de la BBDD para el entrenamiento.")
    parser.add_argument("--pretrain",action="store_true",default=False, help="Preentrena capas del modelo fundacional KEEP.")
    parser.add_argument("--random_subset",action="store_true",default=False, help="Entrenar en cada epoca con un subset aleatorio de los datos.")
    parser.add_argument("--extractor", type=str,default='keep', help="Modelos de extracción de caracteristicas ('keep','cnn').")
    parser.add_argument("--model", type=str,default='mlp', help="Modelos de clasificacion ('graph','mlp','mean_aggre').")

    parser.add_argument("--results_path", type=str, default='/workspace/results', help="Ruta donde se guardaran los resultados")

    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño de batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--patience", type=int, default=20, help="Paciencia para early stopping")

    parser.add_argument("--seed", type=int, default=98, help="Semilla aleatoria")
    parser.add_argument("--num_workers", type=int, default=2, help="Numero de paralelizaciones")

    parser.add_argument("--multilabel_prob_threshold", default=0.4, type=float)
    parser.add_argument("--tumor_abs_thresh", default=0.15, type=float)
    parser.add_argument("--alpha_pe", default=0.01, type=float)
    parser.add_argument("--alpha_pc", default=0.01, type=float)

    return parser.parse_args()

# =================== GENERAL CONFIGURACION ===================
print(" Starting General Configuration ...")
CUDA_LAUNCH_BLOCKING=1
use_amp = torch.cuda.is_available()  # mixed precision if GPU available
args = parse_args()

# Multi-GPU accelerator init
accelerator = Accelerator()
device = accelerator.device
print(f" Usando dispositivo: {device}", flush = True)
torch.cuda.empty_cache()
set_multiple_seed(args.seed)

# =================== PATHS DEFINITION ===================
print(" Defining Paths ...")
pretrain= "_pretrain" if args.pretrain==True else ""
random_subset = "_random_subset" if args.random_subset==True else ""
small_dataset = "_small_dataset" if args.small_dataset==True else ""
name = f"try_{args.extractor}_{args.model}_lr_{args.learning_rate}_bz_{args.batch_size}_epochs_{args.epochs}{pretrain}{random_subset}_weightdecay{args.weight_decay}{small_dataset}"

results_path = os.path.join(args.results_path, name)
os.makedirs(results_path, exist_ok=True)

paths = {   'output_file': os.path.join(results_path, 'output.txt'),
            'results_path': results_path,
            'train_subset_indices_per_epoch_csv_path': os.path.join(results_path, 'train_subset_indices_per_epoch.csv'),
            'result_best_model_path':os.path.join(results_path, 'best_model'),
            'result_last_model_path': os.path.join(results_path, 'last_model'),
            'log_output_path':os.path.join(results_path, 'train_log_output.txt')}

# =================== MODEL CONFIGURATION ===================
print(" Preparing Model ...")
if args.extractor == 'cnn':
    model = CNNEmbedder(clasification_model = args.model)
elif args.extractor =='keep': 
    model = get_keep_ia3_model(clasification_model = args.model, pretrain=args.pretrain).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# =================== LOADING DATA ===================
print(" Preparing DataSet PANDA ...")

with open(paths['log_output_path'], "w") as f:
    f.write(f"Creating Dataset SICAP \n")
df_train = create_file_label_df(args.data_panda_patches, args.data_panda_path,small_dataset= args.small_dataset, NAME = 'PANDA')

label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
df_train['gleason_primary_mapped'] = df_train['gleason_primary'].map(label_mapping)
df_train['gleason_secondary_mapped'] = df_train['gleason_secondary'].map(label_mapping)
df_train['gleason_score'] = df_train['gleason_score'].astype(int)

# =================== DATA SPLIT ===================
train_val_idx, test_idx = train_test_split(df_train.index, test_size=0.10, stratify=df_train["gleason_score"])
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, stratify=df_train.loc[train_val_idx]["gleason_score"])

# =================== DATALOADERS ===================
dl_kwargs = {"batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.num_workers}
if torch.cuda.is_available():
    dl_kwargs["pin_memory"] = True

inicio__load = time.time() 
print("Loading embeddings for Training ...")
data_train = WSI_Dataset(df_train, train_idx, random_subsets= args.random_subset)
train_loader = DataLoader(data_train, shuffle=True, **dl_kwargs)
print("Loading embeddings for Validation ...")
val_loader = DataLoader(WSI_Dataset(df_train, val_idx), shuffle=False, **dl_kwargs)
print("Loading embeddings for Test ...")
test_loader = DataLoader(WSI_Dataset(df_train, test_idx), shuffle=False, **dl_kwargs)
fin__load = time.time()

time_load = fin__load - inicio__load

with open(paths['log_output_path'], "a") as f:
    f.write(f"Time for loading data : {time_load} \n")

# =================== WEIGHTS CALCULATION ===================
print(" Preparing Class Weights and hyperpatameters ...")
primary_weights = compute_class_weights(df_train['gleason_primary_mapped'])
secondary_weights = compute_class_weights(df_train['gleason_secondary_mapped'])
sum_weights = primary_weights + secondary_weights

# =================== TRAINING PARAMETHERS ===================
print(" Preparing Training hyperparamethers ...")
loss = nn.BCELoss(weight=sum_weights) 

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #weight_decay=1e-2

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer,
                                    mode="min",  # porque usamos val_loss (minimizar)
                                    factor=0.1,  # reduce LR a un 10% cuando no mejora
                                    patience = args.patience,  # cuántas épocas esperar sin mejora
    )
model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler )

best_validation_loss = float("inf")
no_improve = 0
best_y_multilabel_prob_threshold, best_tumor_abs_threshold = 0,0

# =================== TRAINING ===================
print(" Starting Training ...")
for epoch in range(args.epochs):
    print(f"\nEpoch {epoch}")
    with open(paths['output_file'], "a") as f:
        f.write(f"\nEpoch {epoch}\n")
    
    # TRAIN 
    train_metrics, train_loss, _,_,_= run_epoch(model, train_loader, device, accelerator, loss, optimizer, mode='train', args=args)
    if accelerator.is_main_process:
        log_epoch_metrics(epoch, "Train", train_metrics, train_loss, paths['log_output_path'])
        save_metrics_to_csv(train_metrics, epoch=epoch,  filepath= os.path.join(paths['results_path'], "train_train_metrics.csv"))
    
    if args.random_subset:
        current_indices = data_train.get_current_subset_indices()
        if accelerator.is_main_process:
            file_exists = os.path.isfile(paths['train_subset_indices_per_epoch_csv_path'])
            with open(paths['train_subset_indices_per_epoch_csv_path'], mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Epoch','Indices'])
                writer.writerow([epoch, current_indices])
        data_train.update_random_subset()
    # VALIDATION
    val_metrics, val_loss, y_multilabel_prob_threshold, tumor_abs_thresh,_ = run_epoch(model, val_loader, device, accelerator, loss, None, mode='val', args=args)
                                                                                    
    print("Selected thresholds - multilabel_prob_threshold:", y_multilabel_prob_threshold, " tumor_abs_threshold:", tumor_abs_thresh, flush=True)
        
    if accelerator.is_main_process:
        log_epoch_metrics(epoch, "Validation", val_metrics, val_loss, paths['log_output_path'])
        save_metrics_to_csv(val_metrics, epoch=epoch,  filepath= os.path.join(paths['results_path'], "train_validation_metrics.csv"))
        
    scheduler.step(val_loss)
    
    #  Early Stopping / Checkpoint
    accelerator.wait_for_everyone()
    if val_loss < best_validation_loss:
        best_validation_loss, no_improve = val_loss, 0
        best_y_multilabel_prob_threshold, best_tumor_abs_threshold = y_multilabel_prob_threshold, tumor_abs_thresh
        accelerator.save_state(paths['result_best_model_path'])
        if accelerator.is_main_process:
            print(f"Mejora de rendimiento. Mejor modelo guardado en {paths['result_best_model_path']} con val_loss: {best_validation_loss:.4f}", flush=True)
            with open(paths['log_output_path'], "a") as f:
                f.write(f"Mejora de rendimiento. Mejor modelo guardado en {paths['result_best_model_path']} con val_loss: {best_validation_loss:.4f}\n")
    else:
        no_improve += 1
        if no_improve >= args.patience:
            if accelerator.is_main_process:
                print('Early stopping activado - Entrenamiento finalizado ', flush=True)
                with open(paths['log_output_path'], "a") as f:
                    f.write(f"Early stopping activado - Entrenamiento finalizado\n")
            break

    # Saving last model
    accelerator.wait_for_everyone()
    accelerator.save_state(paths['result_last_model_path'])

# =================== TEST ===================
print(f' FINALIZADO EL ENTRENAMIENTO ')
# Cargar el estado completo guardado por accelerate, result_best_model_path es la carpeta creada por save_state

if accelerator.is_main_process:
    print("----------------- ENTRENAMIENTO FINALIZADO -----------------",flush=True)
    print("Cargando el mejor modelo para la evaluación en el conjunto de test...", flush=True)
    print(f"  UMBRALES para Test: Y_prob_thresh{best_y_multilabel_prob_threshold} y  tumor_abs_thresh: {best_tumor_abs_threshold:.4f}\n", flush=True)
    with open(paths['log_output_path'], "a") as f:
        f.write("----------------- ENTRENAMIENTO FINALIZADO -----------------\n")
        f.write(f"  UMBRALES para Test: Y_prob_thresh{best_y_multilabel_prob_threshold} y  tumor_abs_thresh: {best_tumor_abs_threshold:.4f}\n")
    
    # Asegurar que el archivo guardado por el proceso principal esté disponible para todos.
    accelerator.wait_for_everyone()

try:
    accelerator.load_state(paths['result_best_model_path'])
    print(f"Checkpoint cargado exitosamente desde {paths['result_best_model_path']}")
except Exception as e:
    print(f"Error cargando checkpoint de accelerate: {e}")

test_metrics, test_loss = run_epoch(model, test_loader, device, accelerator, loss, None, mode='test',args=args, multilabel_prob_threshold=best_y_multilabel_prob_threshold, tumor_abs_thresh=best_tumor_abs_threshold)   

accelerator.wait_for_everyone()
    
if accelerator.is_main_process:
    log_epoch_metrics(epoch, "Test", test_metrics, test_loss, paths['log_output_path'])
    save_metrics_to_csv(test_metrics, epoch=epoch,  filepath= os.path.join(paths['results_path'], "test_metrics.csv"))

# =================== INFERENCE ===================
if args.inference == True: 
        
    with open(paths['output_file'], "a") as f:
        f.write(f"Inference in SICAP \n")
    print(f"Inference in BBDD : SICAP")
        
    df_test = create_file_label_df(args.data_sicap_patches, args.data_sicap_path, NAME = 'SICAP')

    label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
    df_test['gleason_primary_mapped'] = df_test['gleason_primary'].map(label_mapping)
    df_test['gleason_secondary_mapped'] = df_test['gleason_secondary'].map(label_mapping)
    df_test['gleason_score'] = df_test['gleason_score'].astype(int)

    print("Loading embeddings for Inference ...")
    inference_loader = DataLoader(WSI_Dataset(df_test, df_test.index), shuffle=False, **dl_kwargs)

    inference_metrics, inference_loss = run_epoch(model, inference_loader, device, accelerator, loss, None, mode='test',args=args, multilabel_prob_threshold=best_y_multilabel_prob_threshold, tumor_abs_thresh=best_tumor_abs_threshold)   

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        log_epoch_metrics("SICAP dataset", "Test", inference_metrics, inference_loss, paths['log_output_path'])
        save_metrics_to_csv(inference_metrics, epoch=0,  filepath= os.path.join(paths['results_path'], "sicap_test_metrics.csv"))