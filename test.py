import os
import torch
from torch.utils.data import DataLoader

# Módulos propios
from metric_eval import evaluate
from code_slurm.utils import create_file_label_df
from code_slurm.model_2graph import model_graph, weights_init
from code_slurm.data_load import CustomDataset, collate_fn
from safetensors.torch import load_file

# ========================================
# CONFIGURACIÓN
# ========================================
MODEL_TYPE = 3  # 1: Básico, 2: Punctuation, 3: GAT, 4: Mask+GAT, 5: Coords, 6: Coords+GAT
BASE_PATH = '/workspace/code_result/pruebas_slurm_1/prueba_homogenizacion_parches/00001_64_100' #'/workspace/code_result/pruebas_slurm_1/prueba_3' #
batch = 16 

MODEL_CONFIG = {
   
    3: (model_graph, 'SICAP/') #'type3/best_model.pt/'),
    
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# SELECCIÓN DE MODELO
# ========================================
model_fn, model_dir = MODEL_CONFIG.get(MODEL_TYPE, (model_graph, 'other/'))
model = model_fn()

# Paths
sicap_path_embeddings='/workspace/code_result/features_SICAP/embeddings/KEEP'
excel_path_sicap=  '/workspace/code_result/features_SICAP/labels_SICAP.xlsx'

model_path = os.path.join(BASE_PATH, "best_model.pth") #"model.safetensors") 
out_path = os.path.join(BASE_PATH,"metrics_global.txt")
conf_matrix_path = os.path.join(BASE_PATH, "conf_matrix.png")
c_matrix_path = os.path.join(BASE_PATH, "conf_matrix.png")
excel_path = os.path.join(BASE_PATH, "metric_label.xlsx")
c_path = os.path.join(BASE_PATH,  "conf_matrix_SICAP_sum.png")

# PREPARACIÓN DE DATOS

df_test  = create_file_label_df(sicap_path_embeddings,excel_path_sicap)
test_loader_sicap = DataLoader(CustomDataset(df_test, df_test.index),
                            batch_size=batch,
                            shuffle=False, collate_fn=collate_fn)

sicap_loader = DataLoader(test_loader_sicap, batch_size=4, shuffle=False, collate_fn=collate_fn)

# LOAD MODEL
model.load_state_dict(torch.load(model_path, map_location = device) ) #state_dict) #
model.to(device)

avg_loss, acc, prec, rec, f1, kappa, _, _= evaluate(model, sicap_loader, output_path=out_path,  ruta_excel=excel_path, conf_path=c_path, conf_matrix_path2=conf_matrix_path)
