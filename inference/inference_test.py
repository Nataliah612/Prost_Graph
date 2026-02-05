import os
import sys
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader

current_dir = "/workspace/Prost_Graph/"
sys.path.append(os.path.dirname(current_dir))

from models.keep_model import get_keep_ia3_model, weights_init
from utils.train_utils import select_gleason_grades_from_batch_proportions
from utils.data_load import WSI_Dataset,create_file_label_df, collate_fn
from utils.metrics_utils import compute_all_gleason_metrics, log_epoch_metrics, save_metrics_to_csv
from utils.train_utils import   get_model, run_epoch


device = "cuda" if torch.cuda.is_available() else "cpu"

keep_ckpt = "/workspace/results/KEEP/best_model/model.safetensors"
clf_ckpt_peft = "/workspace/results/20260109_212740_model-KEEP_unfrozen_graph_alphape-0_alphapc-0_AdamW_lrate-1e-05_wdecay-1e-05_batch_size-32_epochs-200/best_model/model.safetensors"
clf_ckpt_frozen = "/workspace/results/20260201_065936_model-KEEP_frozen_graph_alphape-0_alphapc-0_AdamW_lrate-1e-05_wdecay-1e-05_batch_size-32_epochs-200/best_model/model.safetensors"

model_peft = get_keep_ia3_model(clasification_model='graph', pretrain = True)
model_frozen = get_keep_ia3_model(clasification_model='graph', pretrain = True)

# ---- cargamos SOLO clasificador + MIL ----
state_dict_peft = load_file(clf_ckpt_peft, device="cpu")
state_dict_frozen = load_file(clf_ckpt_frozen, device="cpu")

KEYWORDS = ("classifier")
filtered_state_peft = {
    k: v for k, v in state_dict_peft.items()
    if any(k.startswith(kw) for kw in KEYWORDS)
}

missing_p, unexpected_p = model_peft.load_state_dict(filtered_state_peft, strict=True) ### False
model_peft.eval().to(device)

KEYWORDS = ("base")
filtered_state_frozen = {
    k: v for k, v in state_dict_frozen.items()
    if any(k.startswith(kw) for kw in KEYWORDS)}

missing_f, unexpected_f = model_frozen.load_state_dict(filtered_state_frozen, strict=False)
model_frozen.eval().to(device)

# =========================
# Inferencia
# =========================

sicap_images_folder_path='/workspace/data/features_SICAP/patches'  #'/workspace/data/TCGA/patches'' #'/workspace/data/features_SICAP/patches' 
sicap_labels_df_path= '/workspace/data/wsi_labels_changed_SICAP.xlsx' # '/workspace/data/datos_PANDAS.xlsx' #  '/workspace/data/wsi_labels_changed_SICAP.xlsx' #'/workspace/data/features_PRAD/tcga_dataset.xlsx' # 

print(f"\n--- SICAP Test Results ---", flush=True)
df_test = create_file_label_df(sicap_images_folder_path, sicap_labels_df_path, NAME = 'SICAP')

label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
df_test['gleason_primary_mapped'] = df_test['gleason_primary'].map(label_mapping)
df_test['gleason_secondary_mapped'] = df_test['gleason_secondary'].map(label_mapping)
df_test['gleason_score'] = df_test['gleason_score'].astype(int)
print("Loading embeddings for Inference ...")
inference_loader = DataLoader(WSI_Dataset(df_test, df_test.index, inference_heatmap=True), shuffle=False, batch_size=32, collate_fn=collate_fn)
        
all_y_true_primary, all_y_true_secondary = [], []
all_y_pred_primary, all_y_pred_secondary = [], []
all_probs, all_y_true_onehot = [],[]
all_wsi_proportions = []

heatmap_list = []
total_loss_sum = 0.0
total_seen = torch.tensor(0, device=device)
with torch.no_grad():
    for inputs, y, _, _ in inference_loader:
        #inputs = inputs.to(device)
        y = y.to(device)

        y_primary = y[:, 0].to(device)
        y_secondary = y[:, 1].to(device)
        y = y.to(device)

        # One-hot multi-label (primary + secondary)
        y_onehot = torch.zeros(y.shape[0], 4, device=device)
        y_onehot.scatter_(1, y, 1)

        lengths = [int(t.size(0)) for t in inputs]
        all_patches = torch.cat([t for t in inputs if t.size(0) > 0], dim=0)
        
        chunks = torch.split(all_patches, 128, dim=0)
        
        all_features =  []

        # 4. Procesar chunk a chunk en GPU
        for chunk in chunks:
            chunk = chunk.to(device)
            features = model_frozen.base.encode_image(chunk)
            all_features.append(features)
        total_feats = torch.cat(all_features, dim=0)
        features_list = list(torch.split(total_feats, lengths, dim=0))
        
        wsi_logits, patch_probs, attention_weights = model_peft.classifier(features_list)

        wsi_proportions = torch.stack([
        p.mean(dim=0) if p.numel() > 0 else torch.zeros(wsi_logits.size(1), device=wsi_logits.device)
        for p in patch_probs], dim=0)  # (B,4)
  
        pred_primary, pred_secondary = select_gleason_grades_from_batch_proportions(wsi_proportions, min_presence_thresh=0.2, secondary_ratio_thresh=0.2 )

        # --- Probabilidades asociadas ---
        probs = wsi_logits.detach().cpu()
        # --- Acumuladores para métricas ---
        all_y_true_primary.append(y_primary.detach().cpu())
        all_y_true_secondary.append(y_secondary.detach().cpu())
        all_y_pred_primary.append(pred_primary.detach().cpu())
        all_y_pred_secondary.append(pred_secondary.detach().cpu())
        all_y_true_onehot.append(y_onehot.detach().cpu())
        all_probs.append(probs)


# Concatenar todo
y_true1 = torch.cat(all_y_true_primary).numpy()
y_true2 = torch.cat(all_y_true_secondary).numpy()
y_pred1 = torch.cat(all_y_pred_primary).numpy()
y_pred2 = torch.cat(all_y_pred_secondary).numpy()
probs = torch.cat(all_probs).numpy()
y_onehot = torch.cat(all_y_true_onehot).numpy()

# --- Calcular métricas ---
metrics = compute_all_gleason_metrics(y_true_primary=y_true1, y_pred_primary=y_pred1,
                                        y_true_secondary=y_true2, y_pred_secondary=y_pred2,
                                        y_true_onehot=y_onehot, y_probs=probs, 
                                        multilabel_prob_threshold = 0.15)

# Inicializar valores opcionales
val_multilabel_prob_threshold = None
val_min_presence_threshold = None
val_secondary_ratio_threshold = None
heatmap_dataframe = None
ckpt_path = "/workspace/results"
log_epoch_metrics("SICAP dataset", "Test_PIXn", metrics, 0, os.path.join(ckpt_path, "inference.txt"))
save_metrics_to_csv(metrics, epoch=0,  filepath= os.path.join(ckpt_path, "inference.csv"))

