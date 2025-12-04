import torch
import torch.nn as nn
   
class model_mlp(nn.Module):
    def __init__(self, mil_aggregation_mode='attentionMIL',chunk_size=512):
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
        
        self.mil = MILArchitecture(classes=[0,1,2,3],aggregation=mil_aggregation_mode, L_value = 512 )
        self.chunk_size = chunk_size 

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
        
        all_features,all_probs  = [], []

        # 4. Procesar chunk a chunk en GPU
        for chunk in chunks:
            # Mover SOLO este trocito a GPU
            chunk_gpu = chunk.to(device, non_blocking=True)

            x_features = self.embedding(chunk_gpu) # EMBEDDING GLEASON GRADE

            x_class = self.classifier(x_features) # CLASIFICATOR GLEASON GRADE
            x_probs = torch.softmax(x_class, dim = 1)      # Activation function 

            # Guardamos los resultados (son vectores pequeños, caben bien en GPU)
            all_features.append(x_features)
            all_probs.append(x_probs.squeeze())

        #patchs_probs_list = list(torch.split(x_probs, lengths, dim=0))
        total_feats = torch.cat(all_features, dim=0) # (N_total, embed_dim)
        total_probs = torch.cat(all_probs, dim=0) 

        wsi_logits = self.mil(total_probs, total_feats, lengths)

        patchs_probs_list = list(torch.split(total_probs, lengths, dim=0))

        return wsi_logits.squeeze(), patchs_probs_list

class Attention(nn.Module):
    def __init__(self, L=512):
        super().__init__()
        D = L // 2
        self.branch_1 = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.branch_2 = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.branch_3 = nn.Linear(D, 1)

    def forward(self, x): # x: (N, L)
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa * xb
        A = self.branch_3(A)    # (N, 1)
        return A.squeeze(-1)    # (N,)
    
class attentionMIL_instance(nn.Module):
    def __init__(self, L_value=512):
        super().__init__()
        self.L = L_value
        self.weight = Attention(L=self.L)

    def forward(self, logits_ws, features_ws):
        """
        logits_ws: (N, num_classes) probabilities/logits per parche de esta WSI
        features_ws: (N, L) embeddings (cls tokens) per parche de esta WSI
        """
        if features_ws.numel() == 0:
            # si no hay parches (caso borde) devolvemos ceros
            return torch.zeros(logits_ws.size(1), device=logits_ws.device)

        weights = self.weight(features_ws)        # (N,)
        attn = torch.softmax(weights, dim=0)      # (N,)
        weighted = logits_ws * attn.unsqueeze(1)  # (N, num_classes)
        w_logits = weighted.sum(dim=0)            # (num_classes,)
        return w_logits

class MILArchitecture(nn.Module):
    def __init__(self, classes, aggregation='max', L_value=None):
        super().__init__()
        self.aggregation = aggregation
        self.L = L_value
        self.nClasses = len(classes)

        if self.aggregation == 'attentionMIL':
            assert self.L is not None, "L_value required for attentionMIL"
            self.attentionModule = attentionMIL_instance(self.L)

    def forward(self, patch_probs, patch_feats, lengths):
        """
        patch_probs: Tensor (N_total, C)  (probabilidades por parche)
        patch_feats: Tensor (N_total, L)  (embeddings cls per parche)
        lengths: list[int] (n_parches por WSI), sum(lengths) == N_total
        Devuelve: Tensor (batch_size, C) con logits/score por WSI
        """
        #assert patch_probs.dim() == 2
        #assert patch_feats.dim() == 2

        # Dividir por WSI
        probs_splits = torch.split(patch_probs, lengths, dim=0)
        feats_splits = torch.split(patch_feats, lengths, dim=0)

        wsis = []
        if self.aggregation == 'max':
            for p in probs_splits:
                if p.size(0) == 0:
                    wsis.append(torch.zeros(self.nClasses, device=patch_probs.device))
                else:
                    wsis.append(torch.max(p, dim=0)[0])
        elif self.aggregation == 'mean':
            for p in probs_splits:
                if p.size(0) == 0:
                    wsis.append(torch.zeros(self.nClasses, device=patch_probs.device))
                else:
                    wsis.append(torch.mean(p, dim=0))
        elif self.aggregation == 'attentionMIL':
            for p, f in zip(probs_splits, feats_splits):
                wsis.append(self.attentionModule(p, f))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        wsi_logits = torch.stack(wsis, dim=0)  
        
        return wsi_logits    