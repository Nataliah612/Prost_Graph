import torch
import pandas as pd
import numpy as np 
import os 
import random
from tqdm import tqdm 
import sys 
import math

from models.keep_model import device
from utils.metrics_utils import compute_all_gleason_metrics, compute_metrics, compute_multilabel_metrics
    
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
    #neg_z = torch.clamp(-z, min=eps)
    branch1 = -torch.log(-z) / t 

    # Rama 2: t*z + const  (const calculado como tensor en mismo device/dtype)
    c_val = -math.log(1.0 / (t ** 2))/t + 1.0 / t
    c = torch.tensor(c_val, device=device, dtype=dtype)
    branch2 = t * z + c

    out = torch.where(mask, branch1, branch2)  
    return out.sum()  

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

def run_epoch(model, dataloader, device, accelerator, 
              loss_BCE, optimizer=None, 
              args=None, mode="train",
              inference_heatmap = False,
              multilabel_prob_threshold=None, tumor_abs_thresh=None,
              alpha_pe=None, alpha_pc=None):
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
        inference_heatmap (bool): si True, devuelve información para poder generar mapas de calor.
        multilabel_prob_threshold (float): umbral de probabilidad para clasificación multilabel. Si es None, se usa args.multilabel_prob_threshold.
        tumor_abs_threshold (float): umbral absoluto de proporción tumoral. Si es None, se usa args.tumor_abs_threshold.
        alpha_pe (float): peso de la restricción Positive Expansion. Si es None, se usa args.alpha_pe.
        alpha_pc (float): peso de la restricción Proportion Constraint. Si es None, se usa args.alpha_pc.

    Returns:
        dict: métricas calculadas.
        float: pérdida media de la época.
    """
    assert mode in ["train", "val", "test"], "mode debe ser 'train', 'val' o 'test'."

    # Acumuladores
    all_y_true_primary, all_y_true_secondary = [], []
    all_y_pred_primary, all_y_pred_secondary = [], []
    all_probs, all_y_true_onehot = [],[]
    all_wsi_proportions = []

    heatmap_list = []
    total_loss_sum = 0.0
    total_seen = torch.tensor(0, device=device)
    
    # Valores de linea de comandos si no se proporcionan como argumentos
    if multilabel_prob_threshold is None or tumor_abs_thresh is None:
        multilabel_prob_threshold = args.multilabel_prob_threshold
        tumor_abs_thresh = args.tumor_abs_thresh

    if alpha_pc is None or alpha_pe is None:
        alpha_pe = args.alpha_pe
        alpha_pc = args.alpha_pc

    # Modo del modelo
    model.train() if mode == "train" else model.eval()

    # Desactivar gradientes si no es entrenamiento
    torch.set_grad_enabled(mode == "train")
    
    for batch_info in tqdm(dataloader, desc=f"{mode.upper()}", disable=not accelerator.is_local_main_process, file=sys.stdout):
        if inference_heatmap:
            inputs, y, patch_id, coords = batch_info
        else:
            inputs, y, _, _= batch_info # inputs: list of tensors [N_patches, C, H, W], y: tensor [batch, 2]

        # Mover etiquetas al device
        y_primary = y[:, 0].to(device)
        y_secondary = y[:, 1].to(device)
        y = y.to(device)

        # One-hot multi-label (primary + secondary)
        y_onehot = torch.zeros(y.shape[0], 4, device=device)
        y_onehot.scatter_(1, y, 1)

        # --- Forward ---
        # (El modelo mueve internamente los inputs al device)
        wsi_prediction, patchs_prediction = model(inputs) # wsi_prediction: Tensor (B,4) , patchs_prediction: list of B tensors, cada uno (N_patches,4), 
                                                                                # weights_tensor_list: list of B tensors, cada uno (N_patches,) o None si no aplica

        # --- Cálculo de proporciones por WSI (media de probabilidades por parche) ---
        # Si algún WSI tuviera 0 parches, usamos vector ceros (Aunque eso debería evitarse)
        wsi_proportions = torch.stack([
            p.mean(dim=0) if p.numel() > 0 else torch.zeros(wsi_prediction.size(1), device=wsi_prediction.device)
            for p in patchs_prediction
        ], dim=0)  # (B,4)

        # print("\n[EPOCH] WSI prediction:", wsi_prediction.shape)
        # for i, p in enumerate(patchs_prediction):
        #     print(f"[EPOCH]  WSI {i} patch preds shape: {p.shape}")
        # print("[EPOCH] wsi_proportions:", wsi_proportions.shape)

        # --- Cálculo de pérdidas ---
        loss = loss_BCE(wsi_prediction, y_onehot)
        loss_pe_pc = loss_restrictions(wsi_proportions, alpha_pe, alpha_pc)
        total_loss = loss + loss_pe_pc

        # --- Backpropagation ---
        if mode == "train":
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Acumulación de pérdidas ---
        batch_sz = len(inputs)
        total_seen += batch_sz
        total_loss_sum += accelerator.gather(total_loss.detach() * batch_sz).sum().item()

        # --- Predicciones (selección de primary/secondary a partir de proporciones) ---
        if mode =='val':
            all_wsi_proportions.append(wsi_proportions.detach().cpu())
                           
        pred_primary, pred_secondary = select_gleason_grades_from_batch_proportions(wsi_proportions, tumor_abs_thresh)

        # --- Probabilidades asociadas ---
        probs = wsi_prediction.detach().cpu()
        # --- Acumuladores para métricas ---
        all_y_true_primary.append(y_primary.detach().cpu())
        all_y_true_secondary.append(y_secondary.detach().cpu())
        all_y_pred_primary.append(pred_primary.detach().cpu())
        all_y_pred_secondary.append(pred_secondary.detach().cpu())
        all_y_true_onehot.append(y_onehot.detach().cpu())
        all_probs.append(probs)

        if inference_heatmap:
            for i in range(batch_sz):
                slide_out = {
                    "slide_id": patch_id[i],
                    "coords": coords[i],
                    "scores": torch.argmax(patchs_prediction[i], dim=1).cpu().numpy(), 
                    #"weights": weights_tensor_list[i]
                    }
                heatmap_list.append(slide_out)

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
        y_true_primary=y_true1, y_pred_primary=y_pred1,
        y_true_secondary=y_true2, y_pred_secondary=y_pred2,
        y_true_onehot=y_onehot, y_probs=probs, multilabel_prob_threshold = multilabel_prob_threshold)
    
     # Inicializar valores opcionales
    val_multilabel_prob_threshold = None
    val_tumor_abs_threshold = None
    heatmap_dataframe = None

    # Ajustes específicos de modo
    if mode == 'val':
        val_metrics, val_multilabel_prob_threshold = select_multilabel_prob_threshold(y_onehot, probs)
        metrics_global, val_tumor_abs_threshold = selecting_abs_threshold(all_wsi_proportions, y_true1, y_true2)
    if inference_heatmap:
        heatmap_dataframe = pd.DataFrame(heatmap_list)
    # --- Return centralizado ---
    return metrics, epoch_loss, val_multilabel_prob_threshold, val_tumor_abs_threshold, heatmap_dataframe
    
def select_multilabel_prob_threshold(y_onehot, probs):
    best_score= -1
    best_multilabel_metrics ={}
    best_multilabel_prob_threshold = None

    thresh_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for multilabel_prob_threshold in thresh_list:
        metrics = compute_multilabel_metrics(
            y_true_onehot=y_onehot,
            y_prob=probs,
            multilabel_prob_threshold = multilabel_prob_threshold)
        score= metrics["f1_macro"] 
        if score> best_score :
            best_score = score
            best_multilabel_metrics = metrics
            best_multilabel_prob_threshold = multilabel_prob_threshold

    return best_multilabel_metrics, best_multilabel_prob_threshold 

def selecting_prob_thresholds(y_true1, y_pred1, y_true2, y_pred2, y_onehot, probs):
    # Primero obtenemos un forward completo

    best_score1= -1
    best_y_prob_threshold = None
    best_metrics ={}
    thresh_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    for y_prob_threshold in thresh_list:

        metrics = compute_all_gleason_metrics(  y_true_primary=y_true1,
                                                y_pred_primary=y_pred1,
                                                y_true_secondary=y_true2,
                                                y_pred_secondary=y_pred2,
                                                y_true_onehot=y_onehot,
                                                y_probs=probs,y_prob_threshold = y_prob_threshold)

        score_f1 = metrics['multilabel']["f1_macro"] 
        
        if score_f1 > best_score1 :
            best_score1 = score_f1
            best_y_prob_threshold = y_prob_threshold
            best_metrics = metrics

    return best_metrics, best_y_prob_threshold 

def selecting_abs_threshold(all_wsi_proportions, y_primary, y_secondary):
    # Concatenar proporciones
    all_proportions = torch.cat(all_wsi_proportions) # Necesitas agregar all_wsi_proportions a tus listas de acumulación.

    best_val_f1 = 0
    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 

    for i in thresh_list:
        pred_primary, pred_secondary = select_gleason_grades_from_batch_proportions(all_proportions, i)
        y_true_global = y_primary + y_secondary
        y_pred_global = pred_primary + pred_secondary
        metrics_global = compute_metrics(y_true_global, y_pred_global)

        if metrics_global['accuracy'] >= best_val_f1:
            best_val_f1 = metrics_global['accuracy']
            tumor_abs_threshold = i
    return metrics_global, tumor_abs_threshold