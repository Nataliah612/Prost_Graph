
import numpy as np 
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import gc
from tqdm import tqdm

def create_file_label_df(embedding_path, data_path, slide_col='image_id'):
    features_list = []
    labels_list = []
    patient_list = []
    gleason_score = []
    gleason_primary_list = []
    gleason_secondary_list = []
    df = pd.read_excel(data_path)
    embeddings = [f for f in os.listdir(embedding_path) if f.endswith('.npy')]

    # Agrega tqdm aquí
    for f in tqdm(embeddings, desc="Procesando embeddings"):
        slide_id = f.split('.npy')[0]
        if slide_col == 'slide_id':
            slide_id = slide_id.split('_')[0]
        row = df[df[slide_col] == slide_id]

        if not row.empty:
            ruta_npy = os.path.join(embedding_path, f)

            gleason = row[['gleason_score']].values
            patient_select = row['image_id'].values
            labels_selected = row[['gleason_primary', 'gleason_secondary']].values
            labels_selected = np.array(labels_selected, dtype=int)[0]

            gleason_score.append(gleason.item())
            features_list.append(ruta_npy)
            labels_list.append(labels_selected)
            patient_list.append(patient_select)
            gleason_primary_list.append(row['gleason_primary'].values[0])
            gleason_secondary_list.append(row['gleason_secondary'].values[0])

    df_train = pd.DataFrame({
        'features': features_list,
        'labels': labels_list,
        'gleason_score': gleason_score,
        'gleason_primary': gleason_primary_list,
        'gleason_secondary': gleason_secondary_list
    })

    return df_train
    
class CustomDataset(Dataset):
    def __init__(self, dataframe, indices):
        """
        Args:
            dataframe (pd.DataFrame): El dataframe que contiene las características y las etiquetas.
            indices (list): Lista de índices que especifican qué filas del dataframe utilizar.
        """
        gc.collect()
        self.dataframe = dataframe
        self.indices = indices  # Almacena los índices proporcionados

        self.features = []
        for i in tqdm(indices, desc="Cargando features"):
            features = self.dataframe.iloc[i]['features']
            features = np.load(features, mmap_mode='r')
            self.features.append(features)

        self.labels = [
            torch.tensor(self.dataframe.iloc[i]['labels'], dtype=torch.float32)
            for i in tqdm(indices, desc="Cargando labels")
        ]

    def __getitem__(self, idx):
        
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.indices)
        
class RandomDataset(Dataset):
    def __init__(self, dataframe, indices,len_subset=1000,small_dataset_for_test = False):
        """
        Args:
            dataframe: pd.DataFrame con una columna 'features' que contiene rutas a .h5
            indices: lista de índices (filas del dataframe) que forman el universo
            len_subset: tamaño del subset aleatorio por época
        """
        gc.collect()
        self.dataframe = dataframe
        self.indices = list(indices) 
        self.len_subset = int(len_subset)

        self.features = []
        self.labels = []

        for i in indices:
            row = self.dataframe.iloc[i]
            label1 = row['gleason_primary']
            label2 = row['gleason_secondary']
            self.labels.append(torch.tensor([label1, label2], dtype=torch.long))
            
            features = np.load(row['features'], mmap_mode='r')
            
            # Aplicamos transform a cada imagen
            features_tensor = torch.stack(features)
            
            self.features.append(features_tensor)

        # Inicializa ind_subset (hay que llamar update_random_list() antes de usar)
        self.ind_subset = None
        self.update_random_list()

    def update_random_list(self):
        """Genera nuevos índices aleatorios para el subset."""

        self.ind_subset = np.random.choice(self.indices, self.len_subset, replace=True) # Generar indices aleatorios 

    def get_current_indices(self):
        """Devuelve la lista de índices usados en esta época, indices originales del dataframe"""

        return self.ind_subset.tolist()
    
    def __len__(self):
        return len(self.ind_subset)
    
    def __getitem__(self, idx):
       
        pos_relative = self.indices.index(self.ind_subset[idx])

        return self.images[pos_relative], self.labels[pos_relative]

def collate_fn(batch):
    features, labels = zip(*batch)
    
    # Rellenar las secuencias de features (si es necesario)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    # Convertir labels a tensor
    labels = torch.stack(labels, dim=0)
    return features_padded, labels

def extractor(dataloader):
    for features, labels in dataloader:
        return features, labels 