import torch 
import numpy as np
import os 
import csv 

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score, cohen_kappa_score, roc_auc_score, average_precision_score)

def compute_metrics(y_true, y_pred):
    """
    Calcula métricas globales de clasificación con promedios micro, macro y weighted.
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
  
    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision
    prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    # Recall
    rec_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # F1-score
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return {
        "accuracy": acc,

        "precision_micro": prec_micro,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,

        "recall_micro": rec_micro,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,

        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,

        "kappa": kappa,
    }

def compute_multilabel_metrics(y_true_onehot, y_prob, y_prob_threshold=0.4):
    """
    Métricas para clasificación multietiqueta.

    y_true_onehot: array (n_samples, n_labels) con 0/1
    y_prob: array (n_samples, n_labels) con probabilidades en [0,1]
    """

    if isinstance(y_true_onehot, torch.Tensor):
        y_true_onehot = y_true_onehot.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    # Predicciones binarias aplicando umbral
    y_pred = (y_prob >= y_prob_threshold).astype(int)

    metrics = {}

    # Accuracy (Exact match)
    metrics["subset_accuracy"] = accuracy_score(y_true_onehot, y_pred)

    # ----- MICRO -----
    metrics["precision_micro"] = precision_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )
    metrics["f1_micro"] = f1_score(
        y_true_onehot, y_pred, average="micro", zero_division=0
    )

    # ----- MACRO -----
    metrics["precision_macro"] = precision_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true_onehot, y_pred, average="macro", zero_division=0
    )

    # ----- WEIGHTED -----
    metrics["precision_weighted"] = precision_score(
        y_true_onehot, y_pred, average="weighted", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(
        y_true_onehot, y_pred, average="weighted", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        y_true_onehot, y_pred, average="weighted", zero_division=0
    )

    # ----- AUROC -----
    try:
        metrics["auroc_macro"] = roc_auc_score(
            y_true_onehot, y_prob, average="macro"
        )
        metrics["auroc_weighted"] = roc_auc_score(
            y_true_onehot, y_prob, average="weighted"
        )
    except ValueError:
        metrics["auroc_macro"] = float("nan")
        metrics["auroc_weighted"] = float("nan")

    # ----- AUPRC -----
    try:
        metrics["auprc_macro"] = average_precision_score(
            y_true_onehot, y_prob, average="macro"
        )
        metrics["auprc_weighted"] = average_precision_score(
            y_true_onehot, y_prob, average="weighted"
        )
    except ValueError:
        metrics["auprc_macro"] = float("nan")
        metrics["auprc_weighted"] = float("nan")

    return metrics



def compute_all_gleason_metrics(y_true_primary, y_pred_primary,
                                y_true_secondary, y_pred_secondary, y_true_onehot=None,y_probs=None, y_prob_threshold=0.4):
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
        metrics["multilabel"] = compute_multilabel_metrics(y_true_onehot, y_probs, y_prob_threshold=y_prob_threshold)

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

    # =======================
    #  Primary / Secondary / Global
    # =======================
    for key in ["primary", "secondary", "global"]:
        m = metrics[key]
        log_str.append(f"\n-- {key.upper()} --")
        log_str.append(f"  Accuracy : {m['accuracy']:.4f}")

        log_str.append(f"  Precision (Micro)   : {m['precision_micro']:.4f}")
        log_str.append(f"  Precision (Macro)   : {m['precision_macro']:.4f}")
        log_str.append(f"  Precision (Weighted): {m['precision_weighted']:.4f}")

        log_str.append(f"  Recall (Micro)      : {m['recall_micro']:.4f}")
        log_str.append(f"  Recall (Macro)      : {m['recall_macro']:.4f}")
        log_str.append(f"  Recall (Weighted)   : {m['recall_weighted']:.4f}")

        log_str.append(f"  F1 Score (Micro)    : {m['f1_micro']:.4f}")
        log_str.append(f"  F1 Score (Macro)    : {m['f1_macro']:.4f}")
        log_str.append(f"  F1 Score (Weighted) : {m['f1_weighted']:.4f}")

        log_str.append(f"  Kappa               : {m['kappa']:.4f}")

    # =======================
    #  Multilabel (si existe)
    # =======================
    if "multilabel" in metrics:
        m_ml = metrics["multilabel"]
        log_str.append(f"\n-- MULTILABEL --")

        log_str.append(f"  Accuracy     : {m_ml['subset_accuracy']:.4f}")

        log_str.append(f"  Precision (Micro)   : {m_ml['precision_micro']:.4f}")
        log_str.append(f"  Precision (Macro)   : {m_ml['precision_macro']:.4f}")
        log_str.append(f"  Precision (Weighted): {m_ml['precision_weighted']:.4f}")

        log_str.append(f"  Recall (Micro)      : {m_ml['recall_micro']:.4f}")
        log_str.append(f"  Recall (Macro)      : {m_ml['recall_macro']:.4f}")
        log_str.append(f"  Recall (Weighted)   : {m_ml['recall_weighted']:.4f}")

        log_str.append(f"  F1 Score (Micro)    : {m_ml['f1_micro']:.4f}")
        log_str.append(f"  F1 Score (Macro)    : {m_ml['f1_macro']:.4f}")
        log_str.append(f"  F1 Score (Weighted) : {m_ml['f1_weighted']:.4f}")

        log_str.append(f"  AUROC (Macro)       : {m_ml['auroc_macro']:.4f}")
        log_str.append(f"  AUROC (Weighted)    : {m_ml['auroc_weighted']:.4f}")

        log_str.append(f"  AUPRC (Macro)       : {m_ml['auprc_macro']:.4f}")
        log_str.append(f"  AUPRC (Weighted)    : {m_ml['auprc_weighted']:.4f}")

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
        metrics (dict): Diccionario con métricas en las keys
                        "primary", "secondary", "global" y opcionalmente "multilabel".
        epoch (int): Número de la época actual.
        filepath (str): Ruta del archivo CSV.
    """

    # =========================
    # Definir columnas base
    # =========================
    fieldnames = ["epoch"]

    for prefix in ["primary", "secondary", "global"]:
        fieldnames.extend([
            f"{prefix}_accuracy",

            f"{prefix}_precision_micro",
            f"{prefix}_precision_macro",
            f"{prefix}_precision_weighted",

            f"{prefix}_recall_micro",
            f"{prefix}_recall_macro",
            f"{prefix}_recall_weighted",

            f"{prefix}_f1_micro",
            f"{prefix}_f1_macro",
            f"{prefix}_f1_weighted",

            f"{prefix}_kappa",
        ])

    # =========================
    # Columnas multilabel
    # =========================
    if "multilabel" in metrics:
        fieldnames.extend([
            "multilabel_subset_accuracy",

            "multilabel_precision_micro",
            "multilabel_precision_macro",
            "multilabel_precision_weighted",

            "multilabel_recall_micro",
            "multilabel_recall_macro",
            "multilabel_recall_weighted",

            "multilabel_f1_micro",
            "multilabel_f1_macro",
            "multilabel_f1_weighted",

            "multilabel_auroc_macro",
            "multilabel_auroc_weighted",

            "multilabel_auprc_macro",
            "multilabel_auprc_weighted",
        ])

    # =========================
    # Aplanar métricas
    # =========================
    row = {"epoch": epoch}

    for prefix in ["primary", "secondary", "global"]:
        m = metrics[prefix]
        row.update({
            f"{prefix}_accuracy": m["accuracy"],

            f"{prefix}_precision_micro": m["precision_micro"],
            f"{prefix}_precision_macro": m["precision_macro"],
            f"{prefix}_precision_weighted": m["precision_weighted"],

            f"{prefix}_recall_micro": m["recall_micro"],
            f"{prefix}_recall_macro": m["recall_macro"],
            f"{prefix}_recall_weighted": m["recall_weighted"],

            f"{prefix}_f1_micro": m["f1_micro"],
            f"{prefix}_f1_macro": m["f1_macro"],
            f"{prefix}_f1_weighted": m["f1_weighted"],

            f"{prefix}_kappa": m["kappa"],
        })

    # =========================
    # Métricas multilabel
    # =========================
    if "multilabel" in metrics:
        m_ml = metrics["multilabel"]
        row.update({
            "multilabel_subset_accuracy": m_ml["subset_accuracy"],

            "multilabel_precision_micro": m_ml["precision_micro"],
            "multilabel_precision_macro": m_ml["precision_macro"],
            "multilabel_precision_weighted": m_ml["precision_weighted"],

            "multilabel_recall_micro": m_ml["recall_micro"],
            "multilabel_recall_macro": m_ml["recall_macro"],
            "multilabel_recall_weighted": m_ml["recall_weighted"],

            "multilabel_f1_micro": m_ml["f1_micro"],
            "multilabel_f1_macro": m_ml["f1_macro"],
            "multilabel_f1_weighted": m_ml["f1_weighted"],

            "multilabel_auroc_macro": m_ml["auroc_macro"],
            "multilabel_auroc_weighted": m_ml["auroc_weighted"],

            "multilabel_auprc_macro": m_ml["auprc_macro"],
            "multilabel_auprc_weighted": m_ml["auprc_weighted"],
        })

    # =========================
    # Guardar en CSV
    # =========================
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def build_heatmap(coords, scores, patch_size=512, scale_factor=16, save_path=None):
    """
    coords: Nx2 array (x, y)
    scores: N array con valores 0,1,2,3
    Produce heatmap en RGB con colores fijos por clase.
    """

    # Escalar coordenadas
    coords = coords // scale_factor
    patch_size = patch_size // scale_factor

    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()

    width  = max_x - min_x + 1
    height = max_y - min_y + 1

    # Ajustar coordenadas
    shifted_coords = coords.copy()
    shifted_coords[:, 0] -= min_x
    shifted_coords[:, 1] -= min_y

    # Heatmap RGB
    heatmap = np.zeros((height + patch_size, width + patch_size, 3), dtype=np.uint8)

    # Mapa de colores
    color_map = {
        3: (255,   0,   0),   # Rojo
        2: (255, 128,   0),   # Naranja
        1: (255, 255,   0),   # Amarillo
        0: (  0,   0, 255)    # Azul
    }

    # Dibujar los parches según su score en color sólido
    for (x, y), s in zip(shifted_coords, scores):
        color = color_map.get(int(s), (255, 255, 255))  # Blanco si algo raro
        heatmap[y:y+patch_size, x:x+patch_size] = color

    img = Image.fromarray(heatmap, mode="RGB")

    if save_path:
        img.save(save_path)

    return img

from PIL import Image, ImageDraw


def overlay_heatmap_on_image(orig, heatmap, alpha=0.45):
    """Devuelve el overlay del heatmap sobre la imagen original."""
    if heatmap.mode != "RGB":
        heatmap = heatmap.convert("RGB")
    
    heatmap_resized = heatmap.resize(orig.size)
    # Asegurar que el heatmap sea RGB
    
    return Image.blend(orig, heatmap_resized, alpha)

def draw_gt_patches(orig, df_gt, slide_id, patch_size=512, scale_factor=16):
    """
    Colorea los parches GT sobre la imagen original con transparencia.
    """
    alpha = 80
    # Colores clase → RGB
    COLOR_MAP = {
        5: (255, 0, 0, alpha),      # Rojo
        4: (255, 140, 0, alpha),    # Naranja
        4.5: (255, 140, 0, alpha),    # Naranja
        3: (255, 255, 0, alpha),    # Amarillo
        0: (0, 0, 255, alpha),      # Azul
    }
    slide_id = slide_id.split('.')[0]
    df_slide = df_gt[df_gt["Slide_ID"] == slide_id]

    # Convertir imagen a RGBA para permitir alpha
    orig = orig.convert("RGBA")

    # Capa transparente donde dibujar los colores
    overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    ps = patch_size // scale_factor

    for _, row_gt in df_slide.iterrows():
        x = int(row_gt["X"] // scale_factor)
        y = int(row_gt["Y"] // scale_factor)
        val = int(row_gt["Valor_Mask"])

        if val not in COLOR_MAP:
            continue
        
        color = COLOR_MAP[val]  # rgba con alpha

        # Dibujar parche transparente
        draw.rectangle([x, y, x + ps, y + ps], fill=color)

    # Combinar overlay de parches con la imagen original
    blended = Image.alpha_composite(orig, overlay)

    return blended.convert("RGB")
