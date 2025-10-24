import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.utils import FocalLoss

# --------------------- plotting helpers (sin cambios lógicos) ---------------------
def plot_learning_model(epochs, train_losses, val_losses, train_accuracies, val_accuracies, plot_lr_path):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()

    plt.savefig(plot_lr_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_lr_model_opt(epochs, train_losses, train_accuracies, plot_lr_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()

    plt.savefig(plot_lr_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------- métricas por clase optimizada ---------------------
def calcular_metricas_por_clase(y_true, y_pred, labels=None, ruta="/workspace/graph_model_basic/"):
    """
    Devuelve un DataFrame con métricas por clase (Accuracy, Precision, Recall, F1).
    Se aprovecha precision/recall/f1 con average=None para obtener vectores por clase.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Accuracy por clase (accuracy binaria por clase)
    accuracies = []
    for lab in labels:
        y_true_bin = (y_true == lab).astype(int)
        y_pred_bin = (y_pred == lab).astype(int)
        accuracies.append(accuracy_score(y_true_bin, y_pred_bin))

    df_metricas = pd.DataFrame({
        "Clase": labels,
        "Accuracy": accuracies,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1-Score": f1_per_class
    })
    return df_metricas, ruta


def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    k = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return accuracy, precision, recall, f1, k


def _compute_global_metrics(y_true, y_pred, labels, ruta_excel):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    df = pd.DataFrame({
        "Class": labels,
        "Precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "F1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
    })
    df.to_excel(ruta_excel, index=False)
    print(f"Métricas por clase guardadas en {ruta_excel}")
    return acc, prec, rec, f1, kappa, df


# --------------------- evaluate optimizada ---------------------
def evaluate(
    model,
    dataloader,
    output_path,
    ruta_excel,
    conf_path,
    conf_matrix_path2,
    label_mapping={0: 0, 3: 1, 4: 2, 5: 3},
):
    """
    Versión optimizada de evaluate:
    - Evita conversiones repetidas CPU<->GPU en el loop.
    - Usa mapeos vectorizados mediante tensores de mapeo.
    - Agrupa resultados con torch.cat (menos Python list overhead).
    - Mantiene mismo comportamiento de métricas/plots.
    """

    model.eval()
    # Un único loss (ambos son CrossEntropyLoss en tu ejemplo)
    loss_fn1 = FocalLoss()
    loss_fn2 = FocalLoss()

    total_loss = 0.0
    sample_count = 0

    # colecciones en tensors
    gs_true_list = []
    gs_pred_list = []
    y1_list = []
    y2_list = []
    out1_list = []
    out2_list = []

    with torch.no_grad():
       
        inverse_label_mapping={0:0, 1:3, 2:4, 3:5}
        for inputs, y in dataloader:
            inputs, y = inputs.to(device), y.to(device)

            y1 = torch.tensor([label_mapping[i.item()] for i in y[:, 0]], device=device)
            y2 = torch.tensor([label_mapping[i.item()] for i in y[:, 1]], device=device)

            out1, out2 = model(inputs, y)

            # --------- pérdidas (opcional) ----------
            loss = (loss_fn1(out1, y1) + loss_fn2(out2, y2))
            batch_sz   = inputs.size(0)
            total_loss += loss.item() * batch_sz
            sample_count += batch_sz

            # --------- Gleason score combinado -------
            out1 = torch.tensor([inverse_label_mapping[i.item()]  for i in out1.argmax(1)])
            out2 = torch.tensor([inverse_label_mapping[i.item()]  for i in out2.argmax(1)])

            y1 = torch.tensor([inverse_label_mapping[i.item()]  for i in y1])
            y2 = torch.tensor([inverse_label_mapping[i.item()]  for i in y2])

            gs_true = (y1 + y2).cpu()
            gs_pred = (out1 + out2).cpu()

            
            # Acumular (mantener en GPU hasta el final)
            gs_true_list.append(gs_true)
            gs_pred_list.append(gs_pred)
            y1_list.append(y1)
            y2_list.append(y2)
            out1_list.append(out1)
            out2_list.append(out2)

    # Concatenar y mover a CPU una sola vez
    gs_true_all = torch.cat(gs_true_list).cpu().numpy()
    gs_pred_all = torch.cat(gs_pred_list).cpu().numpy()

    y1_all = torch.cat(y1_list).cpu().numpy()
    y2_all = torch.cat(y2_list).cpu().numpy()
    out1_all = torch.cat(out1_list).cpu().numpy()
    out2_all = torch.cat(out2_list).cpu().numpy()

    avg_loss = total_loss / sample_count

    # ---------- métricas globales ----------
    labels_gs = [0, 6, 7, 8, 9, 10]  # pos. GS combinados
    labels_gs_ = [0, 3, 4, 5]

    acc, prec, rec, f1, kappa, df_cls = _compute_global_metrics(gs_true_all, gs_pred_all, labels_gs, ruta_excel)

    # ---------- matriz de confusión (GS combinado) ----------
    cm = confusion_matrix(gs_true_all, gs_pred_all, labels=labels_gs)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_gs, yticklabels=labels_gs)
    plt.xlabel("Predicted GS")
    plt.ylabel("True GS")
    plt.title("Confusion Matrix (Gleason Score)")
    legend_labels = [
        mpatches.Patch(color="white", label="0 = No canceroso"),
        mpatches.Patch(color="white", label="6,7,8,9 = Canceroso"),
    ]
    plt.legend(handles=legend_labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.5, 1, 1])
    plt.savefig(conf_path, bbox_inches="tight")
    plt.close()

    # ---------- matrices de confusión por grado ----------
    cm1 = confusion_matrix(y1_all, out1_all, labels=labels_gs_)
    cm2 = confusion_matrix(y2_all, out2_all, labels=labels_gs_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", xticklabels=labels_gs_, yticklabels=labels_gs_, ax=axes[0])
    axes[0].set_title("Grado Gleason 1")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Verdadero")

    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", xticklabels=labels_gs_, yticklabels=labels_gs_, ax=axes[1])
    axes[1].set_title("Grado Gleason 2")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Verdadero")

    legend_labels = [
        mpatches.Patch(color="white", label="0 = No canceroso"),
        mpatches.Patch(color="white", label="3,4,5 = Canceroso"),
    ]
    plt.legend(handles=legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0, -0.25))
    plt.tight_layout(rect=[0, 0, 2, 0])
    plt.savefig(conf_matrix_path2, bbox_inches="tight")
    plt.close()

    # ---------- logging global ----------
    with open(output_path, "w") as f:
        f.write("Evaluation (Gleason-Score level)\n")
        f.write(f"Loss      : {avg_loss:.4f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1        : {f1:.4f}\n")
        f.write(f"Kappa     : {kappa:.4f}\n")

    print(
        f"Loss={avg_loss:.4f} | Acc={acc:.4f} | "
        f"Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | κ={kappa:.4f}"
    )

    return acc, prec, rec, f1, kappa

def evaluate(
    model, dataloader, output_path,
    ruta_excel, conf_path, conf_matrix_path2,
    label_mapping={0:0, 3:1, 4:2, 5:3},
):

    model.eval()
    loss_fn1, loss_fn2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss() #, nn.MSELoss()

    total_loss, sample_count = 0.0, 0
    all_gs_true, all_gs_pred, y1_total, y2_total, out1_total, out2_total = [], [],[],[],[],[]

    with torch.no_grad():
        #inverse_label_mapping = {v: (0 if k in [0, 1, 2] else k) for k, v in label_mapping.items()}
        inverse_label_mapping={0:0, 1:3, 2:4, 3:5}
        for inputs, y in dataloader:
            inputs, y = inputs.to(device), y.to(device)

            y1 = torch.tensor([label_mapping[i.item()] for i in y[:, 0]], device=device)
            y2 = torch.tensor([label_mapping[i.item()] for i in y[:, 1]], device=device)

            out1, out2 = model(inputs)

            # --------- pérdidas (opcional) ----------
            loss = (loss_fn1(out1, y1) + loss_fn2(out2, y2))
            batch_sz   = inputs.size(0)
            total_loss += loss.item() * batch_sz
            sample_count += batch_sz

            # --------- Gleason score combinado -------
            out1 = torch.tensor([inverse_label_mapping[i.item()]  for i in out1.argmax(1)])
            out2 = torch.tensor([inverse_label_mapping[i.item()]  for i in out2.argmax(1)])
            y1 = torch.tensor([inverse_label_mapping[i.item()]  for i in y1])
            y2 = torch.tensor([inverse_label_mapping[i.item()]  for i in y2])

            gs_true = (y1 + y2).cpu()
            gs_pred = (out1 + out2).cpu()

            all_gs_true.extend(gs_true.numpy())
            all_gs_pred.extend(gs_pred.numpy())

            y1_total.extend(y1)
            y2_total.extend(y2)
            out1_total.extend(out1)
            out2_total.extend(out2)

    # ───────── métricas globales ─────────
    labels_gs = [0, 6, 7, 8, 9, 10]   # posibles GS combinados (0-6)
    labels_gs_ = [0, 3, 4, 5]   
    acc, prec, rec, f1, kappa, df_cls = _compute_global_metrics(
        all_gs_true, all_gs_pred, labels_gs, ruta_excel
    )
    avg_loss = total_loss / sample_count

    # ───────── matriz de confusión ───────
    cm = confusion_matrix(all_gs_true, all_gs_pred, labels=labels_gs)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_gs, yticklabels=labels_gs)
    plt.xlabel("Predicted GS"); plt.ylabel("True GS"); plt.title("Confusion Matrix (Gleason Score)")
    legend_labels = [
    mpatches.Patch(color="white", label="0 = No canceroso"),
    mpatches.Patch(color="white", label="6,7,8,9 = Canceroso")
]

    plt.legend(handles=legend_labels, loc="lower center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0.5, 1, 1])  # deja espacio abajo para la leyenda
    plt.savefig(conf_path, bbox_inches="tight")
    plt.close()

    # Calcular las matrices de confusión para cada grado
    cm1 = confusion_matrix( y1_total, out1_total,  labels=labels_gs_)
    cm2 = confusion_matrix( y2_total, out2_total,labels=labels_gs_)

    y1_total = np.array(y1_total)
    out1_total = np.array(out1_total)
    
    # Crear una figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Matriz para grado 1
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_gs_, yticklabels=labels_gs_, ax=axes[0])
    axes[0].set_title("Grado Gleason 1")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Verdadero")

    # Matriz para grado 2
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels_gs_, yticklabels=labels_gs_, ax=axes[1])
    axes[1].set_title("Grado Gleason 2")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Verdadero")

    legend_labels = [
    mpatches.Patch(color="white", label="0 = No canceroso"),
    mpatches.Patch(color="white", label="3,4,5 = Canceroso")
]

    plt.legend(handles=legend_labels, loc="lower center", ncol=2, frameon=False,  bbox_to_anchor=(0, -0.25))

    plt.tight_layout(rect=[0, 0, 2, 0])  # deja espacio abajo para la leyenda
    #plt.tight_layout()
    plt.savefig(conf_matrix_path2, bbox_inches="tight")
    plt.close()

    # ───────── logging global ────────────
    with open(output_path, "w") as f:
        f.write("Evaluation (Gleason-Score level)\n")
        f.write(f"Loss      : {avg_loss:.4f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1        : {f1:.4f}\n")
        f.write(f"Kappa     : {kappa:.4f}\n")

    print(
        f"Loss={avg_loss:.4f} | Acc={acc:.4f} | "
        f"Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | κ={kappa:.4f}"
    )

    return acc, prec, rec, f1, kappa