import torch
import torch.nn as nn

class model_mean_aggre(nn.Module):
    def __init__(self, chunk_size=512):
        '''Para utilizar la atencion por parche se requiere de aprender sobre 512 caracteristicas (AttriMIL)'''
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(768, 512), 
            nn.ReLU(True),
            nn.Dropout(0.2))
              
        self.classifier =nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(32, 4))

        self.chunk_size = chunk_size 


    def forward(self, x):
        
        device = next(self.parameters()).device

        # Mover tensores al device
        inputs_list = [t.to(device) for t in x]

        # 1) Concatenar todos los parches del batch
        #mean_features_wsi = [torch.mean(t, dim = 0, keepdim=True) for t in inputs_list]
        lengths = [int(t.size(0)) for t in inputs_list] # Lista de N_patches por WSI, clave para reconstrucción de WSI
        
        if sum(lengths) == 0:
            raise RuntimeError("Batch contains no patches")
        
        all_patches = torch.cat([t for t in inputs_list if t.size(0) > 0], dim=0)  # (N_total, C, H, W)
        chunks = torch.split(all_patches, self.chunk_size, dim=0)
        
        all_features, all_probs = [],[]
        # 4. Procesar chunk a chunk en GPU
        for chunk in chunks:
            # Mover SOLO este trocito a GPU
            chunk_gpu = chunk.to(device, non_blocking=True)

            x_features = self.embedding(chunk_gpu)      # EMBEDDING GLEASON GRADE
            all_features.append(x_features.squeeze())

            x_class = self.classifier(x_features.squeeze())       # CLASIFICATOR GLEASON GRADE
            x_probs = torch.softmax(x_class, dim = 1)   # Activation function 
            
            all_probs.append(x_probs.squeeze())

        # Agregacion de características por la media 
        total_features = torch.cat(all_features, dim=0) 
        patchs_features_list = list(torch.split(total_features, lengths, dim=0))
        wsi_features = torch.stack( [torch.mean(features, dim=0, keepdim=True) for features in patchs_features_list], dim=0) 

        wsi_class = self.classifier(wsi_features)       # CLASIFICATOR GLEASON GRADE
        wsi_probs = torch.sigmoid(wsi_class.squeeze())            # Activation function 

        # Prediccion por parche 
        total_probs = torch.cat(all_probs, dim=0) 
        patchs_probs_list = list(torch.split(total_probs, lengths, dim=0))

        return wsi_probs, patchs_probs_list