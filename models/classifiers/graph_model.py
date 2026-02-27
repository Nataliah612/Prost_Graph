import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class model_graph(nn.Module):
    def __init__(self):
        super(model_graph, self).__init__()
        self.graphgat = gat()
        self.to(device)

    def forward(self, x):
        """ x: tensor (B, N, D) """
        graphs_list = [grafo_fully_connected(xi.squeeze(dim=1), training=self.training) for xi in x]
       
        out, w1, weights_tensor_list = self.graphgat(graphs_list)
        return out, w1, weights_tensor_list



def grafo_fully_connected(x, training = False):
    num_nodos = x.shape[0]
    
    # Creamos todas las combinaciones posibles
    fila = torch.arange(num_nodos).repeat_interleave(num_nodos)
    col = torch.arange(num_nodos).repeat(num_nodos)
    
    # Quitamos autoconexiones 
    mask = fila != col
    edge_index = torch.stack([fila[mask], col[mask]], dim=0)
    
    edge_index, _ = dropout_edge(edge_index, p=0.5, training=training) # Prueba anterior 0.3
        
    return Data(x=x, edge_index=edge_index)

class gat(torch.nn.Module):
    def __init__(self, heads=2): #def __init__(self, heads=4):
        super(gat, self).__init__()
        self.gat1 = GATv2Conv(768, 512, heads=heads, concat=False, dropout=0.5)# con 0.3 overfitting
        self.gat2 = GATv2Conv(512, 64, heads=heads, concat=False,dropout=0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        
       
        self.Aggr = Aggregate_Attnt(L=512, I=1)
        self.classifier = mlp(in_dim=64, out_dim=4)

    def forward(self, graph_list):
        
        #Empaquetar graphs en un solo batch
        batch = Batch.from_data_list(graph_list)
        batch = batch.to(device)

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        lengths = batch.ptr[1:] - batch.ptr[:-1]   # Guardar el numero de parches por WSI 

        # GATConv 
        z1 = self.gat1(x, edge_index) #| (N_parches, 64)
        z1 = F.relu(z1)
        z1 = self.bn1(z1)

        z2 = self.gat2(z1, edge_index)
        z2 = F.relu(z2)
        z2 = self.bn2(z2)

        # Clasificador por patch
        logits = self.classifier(z2) #| (N_parches, 4)
        probs = torch.softmax(logits, dim=1)

        # aggregation por WSI usando batch_idx | Pesos de importancia de los parches 
        w = self.Aggr(z1, batch_idx)

        # Reconstruir patch predictions por WSI
        patch_output = torch.split(probs, lengths.tolist(), dim=0)
        w_output = torch.split(w, lengths.tolist(), dim=0)

        # Weighted per WSI
        wsi_list, weight_list = [],[]

        for p, f in zip(patch_output, w_output):
            f = torch.softmax(f, dim= 0)
            agge = p * f.unsqueeze(1)
            agre_sum = agge.sum(dim=0)
            wsi_list.append(agre_sum) 
            weight_list.append(f)          # (num_classes,)
       
        out_wsi = torch.stack(wsi_list, dim=0)
        
        return out_wsi, patch_output, weight_list

class Aggregate_Attnt(nn.Module):
    def __init__(self, L=512, I=1):
        super().__init__()
        D = L // 2
        self.branch_1 = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.branch_2 = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.branch_3 = nn.Linear(D, I)

    def forward(self, x: Tensor, batch_idx: Tensor):
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa.mul(xb)
        A = self.branch_3(A)
        return A.squeeze(-1) 

class mlp(nn.Module): #Clasificador 
    def __init__(self, in_dim=64, out_dim=4):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.3) #0.1
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)