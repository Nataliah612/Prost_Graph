import os
import torch
from torch.utils.data import DataLoader

# Módulos propios
from utils.metric_eval import evaluate
from utils.data_load import create_file_label_df,CustomDataset, collate_fn
from models.model_1graph import model_gat
from safetensors.torch import load_file

# ========================================
# CONFIGURACIÓN
# ========================================
MODEL_TYPE = 3  
BASE_PATH = '/workspace/results/try_0.001_32_100' 
batch = 16 

MODEL_CONFIG = {
   
    3: (model_gat, 'SICAP/') #'type3/best_model.pt/'),
    
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SELECCIÓN DE MODELO
model_fn, model_dir = MODEL_CONFIG.get(MODEL_TYPE, (model_gat, 'other/'))
model = model_fn()

# Paths
sicap_path_embeddings='/workspace/data/features_SICAP/embeddings/KEEP'
excel_path_sicap=  '/workspace/data/wsi_labels_changed.xlsx'

model_path = os.path.join(BASE_PATH, "best_model.pth") #"model.safetensors") 
out_path = os.path.join(BASE_PATH,"metrics_global.txt")
conf_matrix_path = os.path.join(BASE_PATH, "conf_matrix.png")
c_matrix_path = os.path.join(BASE_PATH, "conf_matrix.png")
excel_path = os.path.join(BASE_PATH, "metric_label.xlsx")
c_path = os.path.join(BASE_PATH,  "conf_matrix_SICAP_sum.png")

# PREPARACIÓN DE DATOS
df_test  = create_file_label_df(sicap_path_embeddings,excel_path_sicap,slide_col='slide_id')
test_loader_sicap = DataLoader(CustomDataset(df_test, df_test.index),
                            batch_size=batch,
                            shuffle=False, collate_fn=collate_fn)

# LOAD MODEL
model.load_state_dict(torch.load(model_path, map_location = device) ) #state_dict) #
model.to(device)

acc, prec, rec, f1, kappa = evaluate(model, test_loader_sicap, output_path=out_path,  ruta_excel=excel_path, conf_path=c_path, conf_matrix_path2=conf_matrix_path)
