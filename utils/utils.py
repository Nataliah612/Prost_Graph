import torch

import numpy as np 
import os 
import random
from tqdm import tqdm 
import sys 
import math
import csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, average_precision_score, roc_auc_score

from torch.cuda.amp import autocast

from models.model_1graph import device


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

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

def compute_class_weights(labels):
    labels = torch.tensor(labels)
    num_classes = 4
    class_counts = torch.bincount(labels, minlength=num_classes)

    weights = 1.0 / (class_counts.float() + 1e-8)
    weights = weights / weights.sum() * num_classes
    
    return weights.to(dtype=torch.float, device=device)

def run_epoch(model, dataloader, device, accelerator, 
              loss_BCE, optimizer=None,
              args=None, mode="train"):
    """
    Ejecuta una época completa (train, val o test) y devuelve métricas y pérdida media.

    Args:
        model: modelo PyTorch.
        dataloader: iterable de batches (inputs, labels).
        device: dispositivo ('cuda' o 'cpu').
        accelerator: objeto de aceleración de HuggingFace.
        loss_BCE: función de pérdida binaria (nn.BCELoss).
        optimizer: optimizador (solo requerido si mode='train').
        args: argumentos con alpha_pe y alpha_pc.
        mode (str): 'train', 'val' o 'test'.

    Returns:
        dict: métricas calculadas.
        float: pérdida media de la época.
    """
    assert mode in ["train", "val", "test"], "mode debe ser 'train', 'val' o 'test'."

    # Acumuladores
    all_y_true_primary, all_y_true_secondary = [], []
    all_y_pred_primary, all_y_pred_secondary = [], []
    all_probs, all_y_true_onehot = [],[]

    total_loss_sum = 0.0
    total_seen = torch.tensor(0, device=device)

    # Modo del modelo
    model.train() if mode == "train" else model.eval()

    # Desactivar gradientes si no es entrenamiento
    torch.set_grad_enabled(mode == "train")

    for inputs, y in tqdm(dataloader, desc=f"{mode.upper()}", disable=not accelerator.is_local_main_process, file=sys.stdout):
        # inputs: list of tensors [N_patches, C, H, W]
        # y: tensor [batch, 2]

        # mover etiquetas al device
        y_primary = y[:, 0].to(device)
        y_secondary = y[:, 1].to(device)
        y = y.to(device)

        # One-hot multi-label (primary + secondary)
        y_onehot = torch.zeros(y.shape[0], 4, device=device)
        y_onehot.scatter_(1, y, 1)

        # --- Forward ---
        # (El modelo mueve internamente los inputs al device)
        
        # dentro del training loop
        with autocast(dtype=torch.float16):
            wsi_prediction, patchs_prediction = model(inputs)

        # --- Cálculo de proporciones por WSI (media de probabilidades por parche) ---
        # Si algún WSI tuviera 0 parches, usamos vector ceros (Aunque eso debería evitarse)
        wsi_proportions = torch.stack([
            p.mean(dim=0) if p.numel() > 0 else torch.zeros(wsi_prediction.size(1), device=wsi_prediction.device)
            for p in patchs_prediction
        ], dim=0)  # (B,4)
    
        # --- Cálculo de pérdidas ---
        loss = loss_BCE(wsi_prediction, y_onehot)
        loss_pe_pc = loss_restrictions(wsi_proportions, args.alpha_pe, args.alpha_pc)
        total_loss = loss + loss_pe_pc

        # --- Backpropagation ---
        if mode == "train":
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

        # --- Acumulación de pérdidas ---
        batch_sz = len(inputs)
        total_seen += batch_sz
        total_loss_sum += accelerator.gather(total_loss.detach() * batch_sz).sum().item()

        # --- Predicciones (selección de primary/secondary a partir de proporciones) ---
        pred_primary, pred_secondary = select_gleason_grades_from_batch_proportions(wsi_proportions)

        # --- Probabilidades asociadas ---
        probs = wsi_prediction.detach().cpu()
        # --- Acumuladores para métricas ---
        all_y_true_primary.append(y_primary.detach().cpu())
        all_y_true_secondary.append(y_secondary.detach().cpu())
        all_y_pred_primary.append(pred_primary.detach().cpu())
        all_y_pred_secondary.append(pred_secondary.detach().cpu())
        all_y_true_onehot.append(y_onehot.detach().cpu())
        all_probs.append(probs)

    # --- Métricas de la época ---
    total_seen = accelerator.reduce(total_seen, reduction="sum").item()
    epoch_loss = total_loss_sum / total_seen

    # Concatenar todo
    y_true1 = torch.cat(all_y_true_primary).numpy()
    y_true2 = torch.cat(all_y_true_secondary).numpy()
    y_pred1 = torch.cat(all_y_pred_primary).numpy()
    y_pred2 = torch.cat(all_y_pred_secondary).numpy()
    probs = torch.cat(all_probs).numpy()
    y_onehot = torch.cat(all_y_true_onehot).numpy()

    # --- Calcular métricas ---
    metrics = compute_all_gleason_metrics(
        y_true_primary=y_true1,
        y_pred_primary=y_pred1,
        y_true_secondary=y_true2,
        y_pred_secondary=y_pred2,
        y_true_onehot=y_onehot,
        y_probs=probs,
    )

    return metrics, epoch_loss

def select_gleason_grades_from_batch_proportions(proportions, tumor_abs_thresh=0.05):
    """
    Selecciona grado primario y secundario a partir de un tensor de proporciones.

    Args:
        proportions: tensor [Batch_size, 4], columnas [p0, p3, p4, p5]
        tumor_abs_thresh: umbral mínimo de probabilidad tumoral para considerar WSI como cáncer

    Returns:
        primary_grade: tensor [Batch_size], grado primario 
        secondary_grade: tensor [Batch_size], grado secundario
    """

    device = proportions.device

    # Extraer proporciones tumorales: columnas 1,2,3 -> p3, p4, p5
    tumor_probs = proportions[:, 1:]  # shape: [Batch_size, 3]
    tumor_classes = torch.tensor([1, 2, 3], device=device)

    # Orden descendente por probabilidad tumoral
    sorted_vals, sorted_idx = torch.sort(tumor_probs, dim=1, descending=True)
    primary_idx = sorted_idx[:, 0]
    secondary_idx = sorted_idx[:, 1]
    primary_vals = sorted_vals[:, 0]
    secondary_vals = sorted_vals[:, 1]

    # Inicialización de grados según las clases tumorales
    primary_grade = tumor_classes[primary_idx]    # shape: [Batch_size]
    secondary_grade = tumor_classes[secondary_idx]  # shape: [Batch_size]

    # Casos donde todas las proporciones tumorales son muy bajas -> benigno
    benign_mask = primary_vals < tumor_abs_thresh
    primary_grade[benign_mask] = 0
    secondary_grade[benign_mask] = 0

    # Patrones secundarios muy débiles -> se iguala al primario
    single_mask = (secondary_vals < tumor_abs_thresh) | (secondary_vals < 0.3 * primary_vals)
    secondary_grade[single_mask] = primary_grade[single_mask]

    return primary_grade, secondary_grade

def loss_restrictions(wsi_proportions: torch.Tensor, alpha_pe: float, alpha_pc: float, t_pe: float = 15.0, t_pc: float = 5.0):
    """
    Calcula la pérdida con restricciones Positive Expansion (PE) y 
    Proportion Constraint (PC) para un batch de vectores de proporciones.

    Args:
        wsi_proportions (torch.Tensor): Tensor [Batch_size, Num_clases].
        alpha_pe (float): Peso de la restricción Positive Expansion.
        alpha_pc (float): Peso de la restricción Proportion Constraint.
        t_pe (float): Parámetro de suavizado para PE.
        t_pc (float): Parámetro de suavizado para PC.

    Returns:
        torch.Tensor: Pérdida total (escalar).
    """
    # Eliminar clase 0 (No Cancer)
    P = wsi_proportions[:, 1:] # shape: [Batch_size, 3]
    B, C = P.shape

    loss_pe = P.new_tensor(0.0)
    loss_pc = P.new_tensor(0.0)

    for b in range(B):
        p = P[b]  # shape: [3]
        # índices de clases con proporción positiva
        pos_idx = torch.nonzero(p > 0.0, as_tuple=False).view(-1)
        nRestrictions = pos_idx.numel()
        if nRestrictions == 0:
            continue

        # Ordenar descendente por proporción
        sorted_vals, sorted_idx = torch.sort(p, descending=True)
        # tomar los primeros nRestrictions
        top_vals = sorted_vals[:nRestrictions]  # [nRestrictions]

        # --- PE: cada restricción es -top_i <= 0  => z1 = -top_i
        z1 = (-top_vals).unsqueeze(-1)  # [nRestrictions, 1]
        loss_pe = loss_pe + log_barrier(z1, t=t_pe)

        # --- PC: comparar top_i y top_{i+1}, para i=0..nRestrictions-2
        if nRestrictions >= 2:
            diffs = -sorted_vals[:nRestrictions - 1] + sorted_vals[1:nRestrictions]  # vector [nRestrictions-1]
            z2 = diffs.unsqueeze(-1)  # [nRestrictions-1, 1]
            loss_pc = loss_pc + log_barrier(z2, t=t_pc)

    return alpha_pe * loss_pe + alpha_pc * loss_pc

def log_barrier(z: torch.Tensor, t: float = 5.0):
    """
       Calcula la pérdida de barrera logarítmica para una restricción z <= 0.
    
    Args:
        z (torch.Tensor): Valor de la restricción.
        t (float): Parámetro de suavizado.
    
    Returns:
        torch.Tensor: Pérdida de barrera logarítmica.
    """
    # Normalizar shape y device
    z = z.view(-1)  
    device = z.device
    dtype = z.dtype

    eps = 1e-12
    thresh = -1.0 / (t ** 2)

    # Máscara (torch.bool)
    mask = z <= thresh

    # Rama 1: -log(-z)/t  (asegurar -z > eps)
    neg_z = torch.clamp(-z, min=eps)
    branch1 = -torch.log(neg_z) / t 

    # Rama 2: t*z + const  (const calculado como tensor en mismo device/dtype)
    c_val = -math.log(1.0 / (t ** 2)) + 1.0 / t
    c = torch.tensor(c_val, device=device, dtype=dtype)
    branch2 = t * z + c

    out = torch.where(mask, branch1, branch2)  
    return out.sum() 

def compute_metrics(y_true, y_pred):
    """
    Calcula métricas globales de clasificación.
    
    Args:
        y_true (array): etiquetas verdaderas
        y_pred (array): etiquetas predichas
    Returns:
        dict: métricas globales (accuracy, precision, recall, f1, kappa)
    """
    # Asegurar que sean numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
  
    # Métricas básicas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "kappa": kappa,
    }

def compute_all_gleason_metrics(y_true_primary, y_pred_primary,
                                y_true_secondary, y_pred_secondary, y_true_onehot=None,y_probs=None):
    """
    Calcula las métricas para Gleason primario, secundario y global.
    
    Args:
        y_true_primary (array/tensor): etiquetas verdaderas del Gleason primario
        y_pred_primary (array/tensor): predicciones del Gleason primario
        y_true_secondary (array/tensor): etiquetas verdaderas del Gleason secundario
        y_pred_secondary (array/tensor): predicciones del Gleason secundario
        y_true_onehot: array (n_samples, n_labels) con 0/1
        y_prob: array (n_samples, n_labels) con probabilidades en [0,1]

    Returns:
        dict: métricas globales, incluyendo primary, secondary, global y multilabel.
    """
    # --- Primary ---
    metrics_primary = compute_metrics(y_true_primary, y_pred_primary)

    # --- Secondary ---
    metrics_secondary = compute_metrics(y_true_secondary, y_pred_secondary)

    # --- Global ---
    y_true_global = y_true_primary + y_true_secondary
    y_pred_global = y_pred_primary + y_pred_secondary
    metrics_global = compute_metrics(y_true_global, y_pred_global)

    # --- Empaquetar resultados en un solo diccionario ---
    metrics = {
        "primary": metrics_primary,
        "secondary": metrics_secondary,
        "global": metrics_global,
    }

     # --- Multilabel ---
    if y_true_onehot is not None and y_probs is not None:
        metrics["multilabel"] = compute_multilabel_metrics(y_true_onehot, y_probs)

    return metrics
def compute_multilabel_metrics(y_true_onehot, y_prob):
    """
    Métricas para clasificación multietiqueta.

    y_true_onehot: array (n_samples, n_labels) con 0/1
    y_prob: array (n_samples, n_labels) con probabilidades en [0,1]
    """

    if isinstance(y_true_onehot, torch.Tensor):
        y_true_onehot = y_true_onehot.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    # Predicciones binarias aplicando umbral 0.5
    y_pred = (y_prob >= 0.5).astype(int) # 0.5 -> 0.4 Para tener mas posititvos aunque no todos sean ciertos. 
    #y_pred = torch.topk(y_prob, k=2)

    metrics = {}

    # Subset accuracy (Exact match)
    metrics["subset_accuracy"] = accuracy_score(y_true_onehot, y_pred)

    # Precision / Recall / F1 (micro = recomendado en multilabel)
    metrics["precision_micro"] = precision_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )
    metrics["f1_micro"] = f1_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )

    # También macro si te interesa
    metrics["precision_macro"] = precision_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )

    # AUROC (promedio macro por etiqueta)
    metrics["auroc_macro"] = roc_auc_score(
        y_true_onehot, y_prob, average="macro"
    )

    # AUPRC (average precision) macro
    metrics["auprc_macro"] = average_precision_score(
        y_true_onehot, y_prob, average="macro"
    )

    return metrics


def log_epoch_metrics(epoch, split_name, metrics, loss, output_path):
    """
    Imprime y guarda métricas ordenadas para cada época, incluyendo la pérdida.

    Args:
        epoch (int): Número de época.
        split_name (str): "Train", "Val" o "Test".
        metrics (dict): Diccionario con claves 'primary', 'secondary', 'global'.
        loss (float): Pérdida promedio de la época.
        output_path (str): Ruta del archivo donde guardar.
    """
    log_str = []
    log_str.append(f"\n=== Época {epoch} | {split_name.upper()} METRICS ===")
    log_str.append(f"Loss global: {loss:.4f}")

    for key in ["primary", "secondary", "global"]:
        m = metrics[key]
        log_str.append(f"\n-- {key.upper()} --")
        log_str.append(f"  Accuracy : {m['accuracy']:.4f}")
        log_str.append(f"  Precision: {m['precision']:.4f}")
        log_str.append(f"  Recall   : {m['recall']:.4f}")
        log_str.append(f"  F1 Score : {m['f1']:.4f}")
        log_str.append(f"  Kappa    : {m['kappa']:.4f}")

     # Añadir sección para métricas multilabel si están presentes
    if "multilabel" in metrics:
        m_ml = metrics["multilabel"]
        log_str.append(f"\n-- MULTILABEL --")
        log_str.append(f"  Subset Accuracy : {m_ml['subset_accuracy']:.4f}")
        log_str.append(f"  Precision (Micro): {m_ml['precision_micro']:.4f}")
        log_str.append(f"  Recall (Micro)   : {m_ml['recall_micro']:.4f}")
        log_str.append(f"  F1 Score (Micro) : {m_ml['f1_micro']:.4f}")
        log_str.append(f"  Precision (Macro): {m_ml['precision_macro']:.4f}")
        log_str.append(f"  Recall (Macro)   : {m_ml['recall_macro']:.4f}")
        log_str.append(f"  F1 Score (Macro) : {m_ml['f1_macro']:.4f}")
        log_str.append(f"  AUROC (Macro)    : {m_ml['auroc_macro']:.4f}")
        log_str.append(f"  AUPRC (Macro)    : {m_ml['auprc_macro']:.4f}")

    # =======================
    #  Imprimir en consola
    # =======================
    print("\n".join(log_str), flush=True)

    # =======================
    #  Guardar en archivo
    # =======================
    with open(output_path, "a") as f:
        f.write("\n".join(log_str) + "\n")


def save_metrics_to_csv(metrics: dict, epoch: int, filepath: str = "metrics.csv"):
    """
    Guarda las métricas de cada época como una fila en un CSV.

    Args:
        metrics (dict): Diccionario con métricas en las keys "primary", "secondary", "global" y "multilabel".
        epoch (int): Número de la época actual.
        filepath (str): Ruta del archivo CSV.
    """

    # Definir el orden de las columnas
    fieldnames = [
        "epoch",
        "primary_accuracy", "primary_precision", "primary_recall", "primary_f1", "primary_kappa", 
        "secondary_accuracy", "secondary_precision", "secondary_recall", "secondary_f1", "secondary_kappa",
        "global_accuracy", "global_precision", "global_recall", "global_f1", "global_kappa"
    ]

    # Añadir columnas para métricas multilabel si están presentes
    if "multilabel" in metrics:
        fieldnames.extend([
            "multilabel_subset_accuracy",
            "multilabel_precision_micro", "multilabel_recall_micro", "multilabel_f1_micro",
            "multilabel_precision_macro", "multilabel_recall_macro", "multilabel_f1_macro",
            "multilabel_auroc_macro", "multilabel_auprc_macro"
        ])

    # Aplanar el diccionario en una fila
    row = {
        "epoch": epoch,
        "primary_accuracy": metrics["primary"]["accuracy"],
        "primary_precision": metrics["primary"]["precision"],
        "primary_recall": metrics["primary"]["recall"],
        "primary_f1": metrics["primary"]["f1"],
        "primary_kappa": metrics["primary"]["kappa"],
      
        "secondary_accuracy": metrics["secondary"]["accuracy"],
        "secondary_precision": metrics["secondary"]["precision"],
        "secondary_recall": metrics["secondary"]["recall"],
        "secondary_f1": metrics["secondary"]["f1"],
        "secondary_kappa": metrics["secondary"]["kappa"],

        "global_accuracy": metrics["global"]["accuracy"],
        "global_precision": metrics["global"]["precision"],
        "global_recall": metrics["global"]["recall"],
        "global_f1": metrics["global"]["f1"],
        "global_kappa": metrics["global"]["kappa"],
    }

    # Añadir valores para métricas multilabel si están presentes
    if "multilabel" in metrics:
        m_ml = metrics["multilabel"]
        row.update({
            "multilabel_subset_accuracy": m_ml["subset_accuracy"],
            "multilabel_precision_micro": m_ml["precision_micro"],
            "multilabel_recall_micro": m_ml["recall_micro"],
            "multilabel_f1_micro": m_ml["f1_micro"],
            "multilabel_precision_macro": m_ml["precision_macro"],
            "multilabel_recall_macro": m_ml["recall_macro"],
            "multilabel_f1_macro": m_ml["f1_macro"],
            "multilabel_auroc_macro": m_ml["auroc_macro"],
            "multilabel_auprc_macro": m_ml["auprc_macro"]
        })

    # Crear el archivo si no existe y escribir cabecera
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)