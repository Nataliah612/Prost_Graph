import torch
import torch.nn as nn

from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from models.classifiers.graph_model import model_graph
from models.classifiers.mlp_model import model_mlp
from models.classifiers.mean_agre_model import model_mean_aggre
   
class CNNEmbedder(nn.Module):
    def __init__(self, clasification_model = 'graph',chunk_size=512):
        '''Para utilizar la atencion por parche se requiere de aprender sobre 512 caracteristicas (AttriMIL)'''
        super().__init__()
        # cargar modelo
   
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
 
        # usar toda la red excepto el classifier final
        self.features = model.features # convolutions
        self.avgpool = model.avgpool    # adaptive avg pooling
        
        self.embedding = nn.Sequential(
            nn.Linear(1280, 768), 
            nn.ReLU(True),
            nn.Dropout(0.2))

        # Envolver con tu modelo final
        if clasification_model == 'graph':
            model_class = model_graph()

        elif clasification_model == 'mlp':
            model_class = model_mlp(mil_aggregation_mode='attentionMIL',chunk_size=512)

        elif clasification_model == 'mean_aggre':
            model_class = model_mean_aggre()
        else:
            print("Error: clasification_model no reconocido")
            
        self.classifier = model_class
        
        self.chunk_size = chunk_size 

        for param in self.features.parameters():
            param.requires_grad = False
        # descongelar solo el último bloque
        for param in self.features[-2].parameters():
            param.requires_grad = True

    def forward(self, x):
        
        device = next(self.parameters()).device

        # Mover tensores al device
        inputs_list = [t.to(device) for t in x]

        # 1) Concatenar todos los parches del batch
        lengths = [int(t.size(0)) for t in inputs_list] # Lista de N_patches por WSI, clave para reconstrucción de WSI
        
        if sum(lengths) == 0:
            raise RuntimeError("Batch contains no patches")
        
        all_patches = torch.cat([t for t in inputs_list if t.size(0) > 0], dim=0)  # (N_total, C, H, W)

        chunks = torch.split(all_patches, self.chunk_size, dim=0)
        
        all_features, features_list = [], []

        # 4. Procesar chunk a chunk en GPU
        for chunk in chunks:
            # Mover SOLO este trocito a GPU
            chunk_gpu = chunk.to(device, non_blocking=True)
            
            # 2) Pasamos los parches por el extractor de caracteristicas
            x_patches = self.features(chunk_gpu)  
            x_patches = self.avgpool(x_patches)
            x_patches = torch.flatten(x_patches, 1)

            x_features = self.embedding(x_patches) # EMBEDDING GLEASON GRADE

            # Guardamos los resultados (son vectores pequeños, caben bien en GPU)
            all_features.append(x_features)

        #patchs_probs_list = list(torch.split(x_probs, lengths, dim=0))
        total_feats = torch.cat(all_features, dim=0) # (N_total, embed_dim)

        features_list = list(torch.split(total_feats, lengths, dim=0))

        wsi_logits, patchs_probs_list, weights = self.classifier(features_list)

        return wsi_logits, patchs_probs_list, weights
    
        