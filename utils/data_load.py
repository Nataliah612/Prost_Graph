import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import h5py
from tqdm import tqdm 
import gc
from PIL import Image
import sys

def create_file_label_df(images_path, data_path, NAME='', slide_col='image_id', small_dataset_for_test = False):
    features_list,patient_list  = [], []
    gleason_score, gleason_primary_list, gleason_secondary_list= [], [],[]

    df = pd.read_excel(data_path)
    df = df.head(150) if small_dataset_for_test else df

    wsis = [f for f in os.listdir(images_path) if f.endswith('.h5')]
    for f in tqdm(wsis, desc="Procesando WSIs", file=sys.stdout):
        slide_id = f.split('.h5')[0]
        if NAME == 'SICAP':
            slide_id = slide_id.split('_')[0]
        row = df[df[slide_col] == slide_id]

        if not row.empty:
            ruta_h5 = os.path.join(images_path, f)
         
            features = ruta_h5
            gleason = row[['gleason_score']].values
            if 'patient_id' in df.columns:
                patient_select=row['patient_id'].values
            else:
                patient_select = row['image_id'].values

            gleason_score.append(gleason.item())
            features_list.append(features)
            patient_list.append(patient_select)
            gleason_primary_list.append(row['gleason_primary'].values[0])
            gleason_secondary_list.append(row['gleason_secondary'].values[0])

    df_train = pd.DataFrame({
        'features': features_list,
        'patients':patient_list,
        'gleason_score': gleason_score,
        'gleason_primary': gleason_primary_list,
        'gleason_secondary': gleason_secondary_list
    })

    return df_train
      
class WSI_Dataset(Dataset):
    """
    Dataset de WSI con opción de precarga completa en CPU o lectura on-demand.

    Args:
        dataframe (pd.DataFrame)
        indices (list[int]): Lista de índices del DataFrame.
        random_subsets (bool): Si True, en cada época se usa un subset aleatorio de tamaño len_subset.
        len_subset (int): Tamaño del subset aleatorio que se usará en cada época.
        load_on_demand (bool): 
            - False => Precarga todas las imágenes en CPU.
            - True  => lectura en __getitem__.
        inference_heatmap (bool): Si True, devuelve también las coordenadas de los parches.
    """
    def __init__(self, dataframe, indices, random_subsets=False, len_subset=1000, load_on_demand=False, inference_heatmap=False):
        gc.collect()
        self.dataframe = dataframe
        self.indices = list(indices)
        self.random_subsets = bool(random_subsets)
        self.load_on_demand = bool(load_on_demand)
        self.inference_heatmap = bool(inference_heatmap)

        # Ajuste del tamaño del subset
        self.len_subset = min(len_subset, len(self.indices)) if random_subsets else len(self.indices)

        # Transformaciones estándar
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224),antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet Normalization Values
                       std=[0.229, 0.224, 0.225])
        ])

        # Inicializar listas de datos
        self.images, self.labels, self.patch_id,  self.coords, self.index = [], [], [], [], []

        # Precarga de imágenes si load_on_demand=False
        if not self.load_on_demand:
            for i in tqdm(self.indices, desc="Loading WSI", file=sys.stdout):
                row = self.dataframe.iloc[i]
                features_path = row['features']
                label = torch.tensor(
                        [row['gleason_primary_mapped'], row['gleason_secondary_mapped']],
                        dtype=torch.long
                )
                with h5py.File(features_path, "r", swmr=True) as h5_file:
                    imgs = h5_file["imgs"][:]
                    if len(imgs.shape)==3:
                        continue
                    coords = h5_file["coords"][:] if inference_heatmap else None
                self.labels.append(label)
                imgs_tensor = torch.stack([self.transform(Image.fromarray(img)) for img in imgs])
                self.images.append(imgs_tensor)
                self.patch_id.append(features_path)
                self.coords.append(coords)
                self.index.append(i)
        else:
            # En modo on-demand solo guardamos rutas
            for i in self.indices:
                row = self.dataframe.iloc[i]
                label = torch.tensor( # Precargar etiquetas (siempre en memoria)
                    [row['gleason_primary_mapped'], row['gleason_secondary_mapped']],
                    dtype=torch.long
                )
            self.feature_paths = [self.dataframe.iloc[i]['features'] for i in self.indices]
            # coords se cargarán desde HDF5 solo si inference_heatmap=True
            self.coords = [None] * len(self.indices)
            self.patch_id = self.feature_paths

        # Mapeo índice original → posición en listas
        self.idx2pos = {orig_idx: pos for pos, orig_idx in enumerate(self.index)}

        # Inicializar subset aleatorio si aplica
        if self.random_subsets:
            self.ind_subset = None
            self.update_random_subset()
        else:
            self.ind_subset = self.index

    def update_random_subset(self):
        """Selecciona un subset aleatorio de WSI para la época actual."""
        if self.random_subsets:
            self.ind_subset = np.random.choice(self.index, self.len_subset, replace=False)

    def get_current_subset_indices(self):
        """Devuelve los índices del subset actual."""
        return list(self.ind_subset)

    def __len__(self):
        return len(self.ind_subset)

    def __getitem__(self, idx):
        """Devuelve (features_tensor, label_tensor) o (features_tensor, label_tensor, patch_id, coords) según los modos."""
        if self.random_subsets:
            orig_idx = int(self.ind_subset[idx])
            pos = self.idx2pos[orig_idx]
        else:
            pos = idx

        label = self.labels[pos]

        # Precarga CPU
        if not self.load_on_demand:
            imgs_tensor = self.images[pos]
            patch_id = self.patch_id[pos]
            coords = self.coords[pos] if self.inference_heatmap else None
        else:
            feature_path = self.feature_paths[pos]
            patch_id = feature_path
            coords = None
            with h5py.File(feature_path, "r") as h5_file:
                imgs = h5_file["imgs"][:]
                if self.inference_heatmap:
                    coords = h5_file.get("coords", None)
            imgs_tensor = torch.stack([self.transform(Image.fromarray(img)) for img in imgs])

        if self.inference_heatmap:
            return imgs_tensor, label, patch_id,  coords
        else:
            return imgs_tensor, label

def collate_fn(batch):
    """
    batch: lista de tuplas:
        - (features_tensor, label_tensor)
        - o (features_tensor, label_tensor, patch_id, coords)
    Devuelve:
        - list_features: lista de tensores por WSI
        - labels: tensor [batch_size, ...]
        - patch_ids: lista (opcional, None si no hay)
        - coords: lista (opcional, None si no hay)
    """
    # Determinar si batch incluye patch_id y coords
    if len(batch[0]) == 2:
        # Normal
        features, labels = zip(*batch)
        return list(features), torch.stack(labels, dim=0), None, None
    elif len(batch[0]) == 4:
        features, labels,patch_ids, coords = zip(*batch)
        return list(features), torch.stack(labels, dim=0), list(patch_ids), list(coords)
    else:
        raise ValueError(f"Batch con estructura inesperada")