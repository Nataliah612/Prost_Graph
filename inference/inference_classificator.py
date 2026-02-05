import os
import sys
import numpy as np
import torch
from safetensors.torch import load_file

current_dir = "/workspace/Prost_Graph/"
sys.path.append(os.path.dirname(current_dir))

from models.keep_model import get_keep_ia3_model
from utils.train_utils import select_gleason_grades_from_batch_proportions

# =========================
# Definición del modelo
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

clf_ckpt = "/workspace/results/20260109_212740_model-KEEP_unfrozen_graph_alphape-0_alphapc-0_AdamW_lrate-1e-05_wdecay-1e-05_batch_size-32_epochs-200/best_model/model.safetensors"

model_peft = get_keep_ia3_model(clasification_model='graph', pretrain = True)

# ---- cargamos SOLO clasificador + MIL ----
state_dict_peft = load_file(clf_ckpt, device="cpu")

KEYWORDS = ("classifier")
filtered_state_peft = {
    k: v for k, v in state_dict_peft.items()
    if any(k.startswith(kw) for kw in KEYWORDS)
}

missing_p, unexpected_p = model_peft.load_state_dict(filtered_state_peft, strict=False) ### False
model_peft.eval().to(device)

# =========================
# Inferencia
# =========================
print(f"\n--- INFERENCE Test Results ---", flush=True)
     
features_list = []

for archivo in sorted(os.listdir('/workspace/results/inference_PIXn/embeddings')):
    if archivo.endswith(".npy"):
        path = os.path.join('/workspace/results/inference_PIXn/embeddings', archivo)
        emb = np.load('/workspace/data/features_SICAP/embeddings/KEEP/18B0016783A_1.npy')#path)
        features_list.append(torch.from_numpy(emb).to(device))

        wsi_logits, patch_probs, attention_weights = model_peft.classifier(features_list)

        wsi_proportions = torch.stack([
        p.mean(dim=0) if p.numel() > 0 else torch.zeros(wsi_logits.size(1), device=wsi_logits.device)
        for p in patch_probs], dim=0)  # (B,4)
  
        pred_primary, pred_secondary = select_gleason_grades_from_batch_proportions(wsi_proportions, min_presence_thresh=0.2, secondary_ratio_thresh=0.2 )
        labels_dic = {0: "NC", 1: "Grade 3", 2: "Grade 4", 3: "Grade 5"}
        pred_primary = labels_dic.get(pred_primary.item())
        pred_secondary = labels_dic.get(pred_secondary.item())
        print(f"Predicted primary grades: {pred_primary}")
        print(f"Predicted secondary grades: {pred_secondary}")
        