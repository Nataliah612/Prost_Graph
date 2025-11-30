
import numpy as np 
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import h5py

from PIL import Image

import gc
from tqdm import tqdm
import sys

def create_file_label_df(images_path, data_path, NAME='',small_dataset=False ,slide_col='image_id'):
    patch_list,patient_list  = [], []
    gleason_score = []
    gleason_primary_list,gleason_secondary_list  = [], []
    df = pd.read_excel(data_path)
    
    df = df.head(170) if small_dataset == True else df

    wsis = [f for f in os.listdir(images_path) if f.endswith('.h5')]
    for f in tqdm(wsis, desc="Procesando WSIs", file=sys.stdout):
        slide_id = f.split('.h5')[0]
        if NAME == 'SICAP':
            slide_id = slide_id.split('_')[0]
        row = df[df[slide_col] == slide_id]

        if not row.empty:
            ruta_h5 = os.path.join(images_path, f)
         
            gleason = row[['gleason_score']].values
            if 'patient_id' in df.columns:
                patient_select=row['patient_id'].values
            else:
                patient_select = row['image_id'].values

            gleason_score.append(gleason.item())
            patch_list.append(ruta_h5)
            patient_list.append(patient_select)
            gleason_primary_list.append(row['gleason_primary'].values[0])
            gleason_secondary_list.append(row['gleason_secondary'].values[0])

    df_train = pd.DataFrame({
        'patch': patch_list,
        'patients':patient_list,
        'gleason_score': gleason_score,
        'gleason_primary': gleason_primary_list,
        'gleason_secondary': gleason_secondary_list
    })

    return df_train
    
class RAM_WSI_Dataset(Dataset):
    """
    Dataset de WSI (Whole Slide Images) con imágenes precargadas en RAM.

    Args:
        dataframe (pd.DataFrame): DataFrame con columnas
            'gleason_primary_mapped', 'gleason_secondary_mapped' y 'features' (ruta al .h5).
        indices (list[int]): Lista de índices del DataFrame que se cargarán en memoria.
        random_subsets (bool): Si True, en cada época se usa un subset aleatorio de tamaño len_subset.
                               Si False, se usa todo el dataset (validación/test).
        len_subset (int): Tamaño del subset aleatorio que se usará en cada época.
                          Si es mayor que el total, se ajusta automáticamente.
    """
    def __init__(self, dataframe, indices, random_subsets=False, len_subset=1000):
        gc.collect()
        self.dataframe = dataframe
        self.indices = list(indices)
        self.random_subsets = bool(random_subsets)

        # Ajustar tamaño del subset
        if self.random_subsets:
            self.len_subset = min(int(len_subset), len(self.indices))
        else:
            self.len_subset = len(self.indices)

        # Transformaciones básicas
        '''self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])'''
        self.transform = T.Compose([
        T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

        # Precargar imágenes y etiquetas en RAM
        self.images, self.labels, valid_indices= [], [], []
        for i in tqdm(self.indices, desc="Loading WSI", file=sys.stdout):
            row = self.dataframe.iloc[i]
            label = torch.tensor(
                [row['gleason_primary_mapped'], row['gleason_secondary_mapped']],
                dtype=torch.long
            )
            self.labels.append(label)

            features_path = row['patch']
            with h5py.File(features_path, "r", swmr=True) as h5_file:
                imgs = h5_file["imgs"][:]  # numpy array [num_imgs, H, W, C]
                if imgs.shape[0]<= 1:
                    continue

            imgs_tensor = torch.stack([self.transform(Image.fromarray(img).convert('RGB')) for img in imgs])
            self.images.append(imgs_tensor)
            valid_indices.append(i)  # agregamos solo los que cargamos

        # Reemplazamos la lista original de índices por los válidos
        self.indices = valid_indices

        # Mapeo índice original → posición en listas cargadas
        self.idx2pos = {orig_idx: pos for pos, orig_idx in enumerate(self.indices)}

        # Inicializar subset
        if self.random_subsets:
            self.ind_subset = None
            self.update_random_subset()
        else:
            self.ind_subset = self.indices

    def update_random_subset(self):
        """Genera un nuevo subset aleatorio (sin reemplazo)."""
        if self.random_subsets:
            self.ind_subset = np.random.choice(self.indices, self.len_subset, replace=False)  

    def get_current_subset_indices(self):
        """Devuelve la lista actual de índices del DataFrame usados en esta época."""
        return list(self.ind_subset)

    def __len__(self):
        return len(self.ind_subset)

    def __getitem__(self, idx):
        """Devuelve las imágenes y etiquetas del elemento idx dentro del subset actual."""
        if self.random_subsets:
            orig_idx = int(self.ind_subset[idx])
            pos = self.idx2pos[orig_idx]
            return self.images[pos], self.labels[pos]
        else:
            return self.images[idx], self.labels[idx]

def collate_fn(batch):
    features, labels = zip(*batch)

    list_features = list(features) # features: tuple de tensores de distinta longitud -> devolvemos como lista
    labels = torch.stack(labels, dim=0) # Convertir labels a tensor
    return list_features, labels

def extractor(dataloader):
    for features, labels in dataloader:
        return features, labels 