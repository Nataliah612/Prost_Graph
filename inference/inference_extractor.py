import os
import sys
from PIL import Image
import numpy as np
import torch
import h5py
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms as T

current_dir = "/workspace/Prost_Graph/"
sys.path.append(os.path.dirname(current_dir))

from models.keep_model import get_keep_ia3_model
from utils.data_load import WSI_Dataset,create_file_label_df, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

clf_ckpt = "/workspace/results/20260201_065936_model-KEEP_frozen_graph_alphape-0_alphapc-0_AdamW_lrate-1e-05_wdecay-1e-05_batch_size-32_epochs-200/best_model/model.safetensors"
model_frozen = get_keep_ia3_model(clasification_model='graph', pretrain = True)

# ---- cargamos SOLO clasificador + MIL ----
state_dict_frozen = load_file(clf_ckpt, device="cpu")

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

features_path =  "/workspace/data/features_SICAP/patches/16B0003388_13.h5"
result_emb_path = "/workspace/results/inference_PIXn/embeddings"

transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224),antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet Normalization Values
                       std=[0.229, 0.224, 0.225])
        ])
with h5py.File(features_path, "r", swmr=True) as h5_file:
    imgs = h5_file["imgs"][:]

imgs_tensor = torch.stack([transform(Image.fromarray(img)) for img in imgs])

print("Loading embeddings for Inference ...")
    
total_loss_sum = 0.0
total_seen = torch.tensor(0, device=device)
with torch.no_grad():
    
    imgs_tensor = imgs_tensor.to(device)
    emb = model_frozen.base.encode_image(imgs_tensor)
    name = features_path.split('/')[-1]
    name = name.split('.')[0]
    path_embedding = os.path.join(result_emb_path, f"{name}.npy")
    emb_np = emb.detach().cpu().numpy()
    print(f"Saving embedding to {path_embedding} ...")
    np.save(path_embedding, emb_np)