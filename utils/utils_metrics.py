import torch

import os 
import csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, average_precision_score, roc_auc_score

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
                                y_true_secondary, y_pred_secondary, y_true_onehot=None,y_probs=None, y_prob_thresh=0.5):
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
        metrics["multilabel"] = compute_multilabel_metrics(y_true_onehot, y_probs, y_prob_thresh)

    return metrics
def compute_multilabel_metrics(y_true_onehot, y_prob, y_prob_thresh = 0.5):
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
    y_pred = (y_prob >= y_prob_thresh).astype(int) # 0.5 -> 0.4 Para tener mas posititvos aunque no todos sean ciertos. 
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

        log_str.append(f"\n-- THRESOLDS --")
        log_str.append(f" y prob threshold : {m_ml['y_prob_threshold']:.4f}")
        log_str.append(f" tumor abs threshold : {m_ml['tumor_abs_threshold']:.4f}")

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