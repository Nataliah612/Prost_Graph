import argparse
import os 
from datetime import datetime
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv

from sklearn.model_selection import GroupShuffleSplit

from accelerate import Accelerator

from utils.data_load import WSI_Dataset,create_file_label_df, collate_fn

from utils.train_utils import set_multiple_seed
from utils.train_utils import   get_model, run_epoch
from utils.metrics_utils import  log_epoch_metrics, save_metrics_to_csv #, compute_metrics_by_class


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL + Patch Classifier")

    parser.add_argument("--panda_img_folder", type=str, default="/workspace/data/pandas/patches", help="Ruta de carpeta con imágenes WSI de Panda en .h5 (Dentro deben contener parches)")
    parser.add_argument("--panda_df_path", type=str, default="/workspace/data/datos_PANDAS.xlsx", help="Ruta del archivo Excel con etiquetas de las WSI de Panda asociadas a la ruta del .h5")

    parser.add_argument("--sicap_img_folder", type=str, default="/workspace/data/features_SICAP/patches", help="Ruta de carpeta con imágenes WSI de SICAP en .h5 (Dentro deben contener parches)")
    parser.add_argument("--sicap_df_path", type=str, default="/workspace/data/wsi_labels_changed_SICAP.xlsx", help="Ruta del archivo Excel con etiquetas de las WSI de SICAP asociadas a la ruta del .h5")

    parser.add_argument("--results_path",type=str, default="/workspace/results", help=" Ruta de resultados") 

    parser.add_argument("--extractor_model", type=str, default="CNN", help="Tipo de modelo encoder para parches ('CNN', 'KEEP')")
    parser.add_argument("--pretrain",action="store_true",help="Descongelar el backbone del encoder para parches")
    parser.add_argument("--classification_mode", type=str, default="graph", help="Modo de funcionamiento del MIL ('graph','mlp','mean_aggre')")

    parser.add_argument("--small_dataset_for_test",action="store_true",help="Usar un dataset pequeño para pruebas")
    parser.add_argument("--load_on_demand",action="store_true",help="Lectura de imagenes en __getitem__. Cuando no esta activo se precargan en __init__.")
    parser.add_argument("--random_subset",action="store_true",help="Uso de subset variable por época de tamaño reducido para agilizar entrenamiento.")

    parser.add_argument("--hiperparam_tuning", action="store_true", help="Activar hiperparametrización (Se probarán combinaciones definidas en el código)")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizador (SGD, AdamW)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum. Solo para SGD")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--weighted_loss",action="store_true",help="Uso de pesos de la distribución de clases en función de perdida")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--patience", type=int, default=20, help="Paciencia para early stopping")
    
    parser.add_argument("--seed", type=int, default=98, help="Semilla aleatoria")

    parser.add_argument("--alpha_pe", type=float, default=0, help="Peso de la restricción Positive Expansion")
    parser.add_argument("--alpha_pc", type=float, default=0, help="Peso de la restricción Proportion Constraint")
    parser.add_argument("--multilabel_prob_threshold", type=float, default=0.4,  help="Umbral de probabilidad para clasificación multilabel.")
    parser.add_argument("--min_presence_threshold", type=float, default=0.1,  help="Umbral mínimo de presencia tumoral para considerar WSI como cáncer.")
    parser.add_argument("--secondary_ratio_threshold", type=float, default=0.3,  help="Umbral mínimo de proporción relativa del secundario respecto al primario.") 

    parser.add_argument("--resume_training", action="store_true", help="Reanudar entrenamiento desde un modelo preentrenado")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Ruta del directorio con el modelo preentrenado (+ Optimizer, Scheduler)")

    parser.add_argument("--experiment_nickname",type=str, default="", help="Apodo para identificar el experimento en la carpeta de resultados") 

    return parser.parse_args()

def main():
    args = parse_args()
    # Multi-GPU accelerator init
    accelerator = Accelerator()
    device = accelerator.device

    print(f"Usando dispositivo: {device}", flush = True)
    torch.cuda.empty_cache()

    set_multiple_seed(args.seed)

    print("Creating PANDA Train Dataset ...", flush = True)
    df_train  = create_file_label_df(args.panda_img_folder, args.panda_df_path, small_dataset_for_test= args.small_dataset_for_test)
    df_train = df_train.sort_values(by=['patients']).reset_index(drop=True)
    # Sistema de mapeo de etiquetas (Grados Person (0,1,2) serán no cancerosos (0) - (3,4,5) tendrán grados del 1 al 3)
    label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}

    df_train['gleason_primary_mapped'] = df_train['gleason_primary'].map(label_mapping)
    df_train['gleason_secondary_mapped'] = df_train['gleason_secondary'].map(label_mapping)
    # Si quieres enteros
    df_train['gleason_score'] = df_train['gleason_score'].astype(int)
    df_train['patients'] = df_train['patients'].astype(str)
    
    # Spliting Data en Training, Validation and Test 
    # --- 1) Split para TEST (10%) ---
    gss_test = GroupShuffleSplit(test_size=0.10, n_splits=1, random_state=args.seed)
    train_val_idx, test_idx = next(gss_test.split(df_train.index, df_train["gleason_score"], groups=df_train["patients"]))

    # --- 2) Split para VALIDACIÓN (20%) dentro del 90% restante ---
    val_fraction_in_remaining = 0.20 / 0.90  # ≈ 0.2222
    gss_val = GroupShuffleSplit(test_size=val_fraction_in_remaining, n_splits=1, random_state=args.seed)
    train_idx, val_idx = next(gss_val.split(train_val_idx, df_train.loc[train_val_idx]["gleason_score"], groups=df_train.loc[train_val_idx]["patients"]))
    
    inicio__load = time.time() 
    print("Train split ...", flush = True)
    data_train = WSI_Dataset(df_train, train_idx, random_subsets=args.random_subset,load_on_demand=args.load_on_demand)
    print("Validation split ...", flush = True)
    data_val = WSI_Dataset(df_train, val_idx,load_on_demand=args.load_on_demand)
    print("Test split ...", flush = True)
    data_test = WSI_Dataset(df_train, test_idx,load_on_demand=args.load_on_demand)
    fin__load = time.time()

    # Comprobar si se debe hacer inferencia con SICAP dataset
    sicap_data = None
    sicap_inference = args.sicap_img_folder is not None and args.sicap_df_path is not None
    if sicap_inference:
        print("Creating SICAP Test Dataset ...", flush=True)
        df_test = create_file_label_df(args.sicap_img_folder,args.sicap_df_path,NAME='SICAP')
        df_test['gleason_primary_mapped'] = df_test['gleason_primary'].map(label_mapping)
        df_test['gleason_secondary_mapped'] = df_test['gleason_secondary'].map(label_mapping)
        sicap_data = WSI_Dataset(df_test, df_test.index,load_on_demand=args.load_on_demand)
        

    load_time = fin__load - inicio__load
    print(f"Tiempo de carga en memoria de las WSI: {load_time:.4f} segundos", flush = True)
    return data_train, data_val, data_test, sicap_data, accelerator, device, args
    

def run_experiment(data_train, data_val, data_test, sicap_data, accelerator, device, args):
    # DataLoaders
    print("Creating DataLoaders ...", flush = True)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(data_val,batch_size=args.batch_size,shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(data_test, batch_size=args.batch_size,shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Inicializamos los modelos
    print("Loading models ...", flush = True)
    model = get_model(args.extractor_model,args.classification_mode,args.pretrain).to(device)

    # Crear carpeta de resultados
    run_results_path = "" 
    if accelerator.is_main_process:
        print("Creating results paths ...", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_nickname = f"{args.experiment_nickname}_" if args.experiment_nickname != "" else ""
        freeze_status = "pretrain" if args.pretrain else "frozen"
        optimizer_name = args.optimizer if args.optimizer =="AdamW" else f"{args.optimizer}_momentum-{args.momentum}"
        # weighted_loss = "_weighted_loss" if args.weighted_loss else ""
        small_dataset = "_small_dataset" if args.small_dataset_for_test else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{experiment_nickname}model-{args.extractor_model}_{freeze_status}_{args.classification_mode}_alphape-{args.alpha_pe}_alphapc-{args.alpha_pc}_{optimizer_name}_lrate-{args.learning_rate}_wdecay-{args.weight_decay}_batch_size-{args.batch_size}_epochs-{args.epochs}{small_dataset}"
        run_results_path = os.path.join(args.results_path, run_name)
        os.makedirs(run_results_path, exist_ok=True)

    print(f"Los resultados se guardarán en: {run_results_path}",flush=True)
    result_best_model_path = os.path.join(run_results_path, 'best_model')
    result_last_model_path = os.path.join(run_results_path, 'last_model')
    log_output_path = os.path.join(run_results_path, 'train_log_output.txt')

    # ------------ Pérdidas y optimizadores ------------------------
    # # Calculamos los pesos 
    # class_weights_primary = compute_class_weights(df_train['gleason_primary_mapped'])
    # class_weights_secondary = compute_class_weights(df_train['gleason_secondary_mapped'])
    # final_weights = (class_weights_primary + class_weights_secondary) / 2

    # if args.weighted_loss:
    #     loss_BCE = nn.BCELoss(pos_weight=final_weights.to(device))
    # else:
    #     loss_BCE = nn.BCELoss()
    loss_BCE = nn.BCELoss()

    # Optimizador
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizador no soportado: {args.optimizer}")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",  # porque usamos val_loss (minimizar)
        factor=0.5,  # reduce LR a un 50% cuando no mejora
        patience = 8,  # cuántas épocas esperar sin mejora
        min_lr=1e-6,
        threshold=1e-3,
        verbose=True  # imprime cambios de LR
    )
    # Multi-GPU accelerator loading
    model, optimizer,scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader, test_loader)

    # Si resume_training está activado, cargar checkpoint
    if args.resume_training:
        if not args.pretrained_model_path:
            raise ValueError("--resume_training requiere especificar --pretrained_model_path")
        accelerator.load_state(args.pretrained_model_path)
        if accelerator.is_main_process:
            print(f"Training from  accelerator checkpoint: {args.pretrained_model_path}", flush = True)
            with open(log_output_path, "a") as f:
                f.write(f"Training from accelerator checkpoint: {args.pretrained_model_path}\n")

    # Early Stopping variables
    best_validation_loss, no_improve = float("inf"), 0
    best_val_multilabel_prob_threshold, best_min_presence_threshold, best_secondary_ratio_threshold = 0, 0, 0

    if args.random_subset:
        # Definir ruta para almacenar información sobre los indices que se usarán en cada época de entrenamiento
        train_subset_indices_per_epoch_csv_path = os.path.join(run_results_path, 'training_subset_indices.csv')

    for epoch in range(args.epochs):
        inicio_ep = time.time()  # Marca el inicio de la época
        if accelerator.is_main_process:
            print(f"\n ----------------- Epoch {epoch} -----------------", flush = True)
        # ============= TRAINING =============
        train_metrics, train_loss, _, _, _, _ = run_epoch(
            model, train_loader, device, accelerator,
            loss_BCE, optimizer,
            args, mode="train")

        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Train", train_metrics, train_loss, log_output_path)
            save_metrics_to_csv(train_metrics, epoch=epoch,  filepath= os.path.join(run_results_path, "train_train_metrics.csv"))
        
        if args.random_subset:
            current_indices = data_train.get_current_subset_indices()
            if accelerator.is_main_process:
                file_exists = os.path.isfile(train_subset_indices_per_epoch_csv_path)
                with open(train_subset_indices_per_epoch_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Epoch','Indices'])
                    writer.writerow([epoch, current_indices])
            data_train.update_random_subset()

        # ============= VALIDATION =============
        val_metrics, val_loss, val_multilabel_prob_threshold, val_min_presence_threshold, val_secondary_ratio_threshold, _ = run_epoch(
            model, val_loader, device, accelerator,
            loss_BCE, None,
            args, mode="val")
        
        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Validation", val_metrics, val_loss, log_output_path)
            save_metrics_to_csv(val_metrics, epoch=epoch,  filepath= os.path.join(run_results_path, "train_validation_metrics.csv"))
        
        scheduler.step(val_loss)

        #  Early Stopping / Checkpoint
        accelerator.wait_for_everyone()
        if val_loss < best_validation_loss:
            best_validation_loss, no_improve = val_loss, 0
            best_val_multilabel_prob_threshold, best_min_presence_threshold, best_secondary_ratio_threshold  = val_multilabel_prob_threshold, val_min_presence_threshold, val_secondary_ratio_threshold
            accelerator.save_state(result_best_model_path)
            if accelerator.is_main_process:
                print(f"Mejora de rendimiento. Mejor modelo guardado en {result_best_model_path} con val_loss: {best_validation_loss:.4f}", flush=True)
                print(f"  UMBRALES para Test OPTIMIZADOS: Multilabel_prob_threshold {best_val_multilabel_prob_threshold} - min_presence_threshold: {best_min_presence_threshold} - secondary_ratio_threshold: {best_secondary_ratio_threshold}\n", flush=True)
                with open(log_output_path, "a") as f:
                    f.write(f"Mejora de rendimiento. Mejor modelo guardado en {result_best_model_path} con val_loss: {best_validation_loss:.4f}\n")
                    f.write(f"  UMBRALES para Test OPTIMIZADOS: Multilabel_prob_threshold {best_val_multilabel_prob_threshold} - min_presence_threshold: {best_min_presence_threshold} - secondary_ratio_threshold: {best_secondary_ratio_threshold}\n")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                if accelerator.is_main_process:
                    print('Early stopping activado - Entrenamiento finalizado ', flush=True)
                    with open(log_output_path, "a") as f:
                        f.write(f"Early stopping activado - Entrenamiento finalizado\n")
                break

        fin_ep = time.time() # Marca el final de la época

        # Saving last model
        accelerator.wait_for_everyone()
        accelerator.save_state(result_last_model_path)

        print(f"Tiempo transcurrido en epoca {epoch}: {fin_ep - inicio_ep :.4f} segundos", flush=True)
        with open(log_output_path, "a") as f:
            f.write(f"Tiempo transcurrido en epoca {epoch}: {fin_ep - inicio_ep :.4f} segundos\n")
    
    # ============= TEST ============= 
    if accelerator.is_main_process:
        print("----------------- ENTRENAMIENTO FINALIZADO -----------------",flush=True)
        print("Cargando el mejor modelo para la evaluación en el conjunto de test...", flush=True)
        print(f"  UMBRALES para Test: Multilabel_prob_threshold {best_val_multilabel_prob_threshold} - min_presence_threshold: {best_min_presence_threshold} - secondary_ratio_threshold: {best_secondary_ratio_threshold}\n", flush=True)
        with open(log_output_path, "a") as f:
            f.write("----------------- ENTRENAMIENTO FINALIZADO -----------------\n")
            f.write(f"  UMBRALES para Test: Multilabel_prob_threshold {best_val_multilabel_prob_threshold} - min_presence_threshold: {best_min_presence_threshold} - secondary_ratio_threshold: {best_secondary_ratio_threshold}\n")
    
    # Asegurar que el archivo guardado por el proceso principal esté disponible para todos.
    accelerator.wait_for_everyone()
    
    # Cargar el estado completo guardado por accelerate, result_best_model_path es la carpeta creada por save_state
    try:
        accelerator.load_state(result_best_model_path)
        print(f"Checkpoint cargado exitosamente desde {result_best_model_path}")
    except Exception as e:
        print(f"Error cargando checkpoint de accelerate: {e}")
    
    # Test
    test_metrics, test_loss, _, _, _, _ = run_epoch(
        model, test_loader, device, accelerator,
        loss_BCE, None,
        args, mode="test",
        multilabel_prob_threshold=best_val_multilabel_prob_threshold, min_presence_threshold=best_min_presence_threshold, secondary_ratio_threshold=best_secondary_ratio_threshold)
    
    if accelerator.is_main_process:
        log_epoch_metrics("Test", "Test", test_metrics, test_loss, log_output_path)
        save_metrics_to_csv(test_metrics, epoch=0,  filepath= os.path.join(run_results_path, "train_test_metrics.csv"))

    # ============= SICAP TEST ============= 
    if sicap_data is not None:
        sicap_test_loader = DataLoader(sicap_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        print(f"\n--- SICAP Test Results ---", flush=True)
        external_test_metrics, external_test_loss, _, _, _, _ = run_epoch(
            model, sicap_test_loader, device, accelerator,
            loss_BCE, None,
            args, mode="test",
            multilabel_prob_threshold=best_val_multilabel_prob_threshold, min_presence_threshold=best_min_presence_threshold, secondary_ratio_threshold=best_secondary_ratio_threshold)
        
        if accelerator.is_main_process:
            log_epoch_metrics("SICAP dataset", "Test", external_test_metrics, external_test_loss, log_output_path)
            save_metrics_to_csv(external_test_metrics, epoch=0,  filepath= os.path.join(run_results_path, "sicap_test_metrics.csv"))

if __name__ == "__main__":
    data_train, data_val, data_test, sicap_data, accelerator, device, args = main()
    if not args.hiperparam_tuning:
        run_experiment(data_train, data_val, data_test, sicap_data, accelerator, device, args)
    else:
        print("Running multiple experiments ...", flush = True)
        experiments = [
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='graph'),
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='mlp'),
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='mean_aggre'),
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='graph', pretrain = True),
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='mlp', pretrain = True),
            dict(extractor_model='KEEP', optimizer='AdamW',classification_mode='mean_aggre', pretrain = True),
        ]
        for exp in experiments:
            exp_args = copy.deepcopy(args)
            for k, v in exp.items():
                setattr(exp_args, k, v)
            run_experiment(data_train, data_val, data_test, sicap_data, accelerator, device, exp_args)