import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.nn import GATConv

from torch import Tensor

from torch_geometric.nn.aggr import Aggregation

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

print(torch.cuda.is_available())   # True si hay GPU disponible
print(torch.cuda.device_count())   # Número de GPUs

def build_attention_graph(node_features, adj_model, k=8, exclude_self=True):
    """
    Devuelve un grafo filtrado por atención y los pesos continuos de cada arista.

    Parameters
    ----------
    node_features : Tensor  (N, D)
        Embeddings de los nodos.
    adj_model : nn.Module
        MLP que, dado (node_i, node_j), produce un peso escalar ∈ (0,1).
    k : int
        Número máximo de vecinos que conserva cada nodo (top-k por atención).
    exclude_self : bool
        Si True, elimina los self-loops antes de seleccionar.

    Returns
    -------
    edge_index : LongTensor  (2, E)
        Índices (source, target) de las aristas seleccionadas.
    edge_attr  : FloatTensor (E,)
        Peso continuo de cada arista, en el mismo orden que edge_index.
    """
    N = node_features.size(0)
    device = node_features.device

    # --- 1) Todas las parejas (dirigido) ----------------------------
    idx = torch.arange(N, device=device)
    row, col = torch.meshgrid(idx, idx, indexing='ij')
    row, col = row.reshape(-1), col.reshape(-1)

    if exclude_self:
        mask = row != col
        row, col = row[mask], col[mask]

    # --- 2) Pesos de atención para cada pareja ----------------------
    node_i, node_j = node_features[row], node_features[col]
    att_weights = adj_model(node_i, node_j).squeeze()          # (E_all,)

    # --- 3) Top-k por nodo -----------------------------------------
    edge_index_list = []
    edge_weight_list = []

    for i in range(N):
        sel = (row == i)
        if sel.sum() == 0:
            continue

        att_i   = att_weights[sel]          # pesos hacia i->*
        col_i   = col[sel]                  # destinos correspondientes

        topk_idx = torch.topk(att_i, min(k, att_i.size(0)), largest=True).indices
        edge_index_list.append(
            torch.stack([torch.full_like(col_i[topk_idx], i),  # source = i
                         col_i[topk_idx]], dim=0)
        )
        edge_weight_list.append(att_i[topk_idx])

    # --- 4) Concatenar resultados ----------------------------------
    edge_index = torch.cat(edge_index_list, dim=1)        # (2, E)
    edge_attr  = torch.cat(edge_weight_list, dim=0)       # (E,)

    return edge_index.long(), edge_attr.float()

class adj(nn.Module):
    def __init__(self):
        super(adj, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        weights_init(self)

    def forward(self, node_i, node_j):
        pair = torch.cat([node_i, node_j], dim=-1)  # (E, 768*2)
        return torch.sigmoid(self.mlp(pair))  # (E, 1)


class Aggregate_Attnt(nn.Module):
    r"""Args:
        L (int): Dimensión de la entrada
        I (int): Dimensión de la salida
    """
    def __init__(self,L=512,I=1):
        super().__init__()
      
        #self.gate_nn = gate_nn
        D = L//2
        self.branch_1 = nn.Sequential(
            nn.Linear(L,D), 
            nn.Tanh())
        self.branch_2 = nn.Sequential(
            nn.Linear(L,D), 
            nn.Sigmoid())
        self.branch_3= nn.Linear(D,I)
        weights_init(self)

    def forward(self, x: Tensor,dim: int = -2): 

        # Weights calculation 
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa.mul(xb)
        A = self.branch_3(A) 

        gate = torch.softmax(A, dim=0) 

        return gate

  
    
class gat(torch.nn.Module):
    def __init__(self, heads=8):
        super(gat,self).__init__()

        self.gat1 = GATConv(768, 256, heads=heads, edge_dim = 1, concat=False)
        self.gat2 = GATConv(256, 64, heads=heads, edge_dim = 1,concat=False)
        #att_net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),nn.Tanh())
        #att_net = Attention()
        self.Aggr = Aggregate_Attnt(L = 768,I = 1)
        self.fc = nn.Linear(64, 4)
       
    def forward(self,data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr,data.batch
        edge_index= edge_index.to(x.device)

        x1, _ = self.gat1(x, edge_index, edge_weight, return_attention_weights=False)
        x1 = F.relu(x1)
        x2, _ = self.gat2(x1, edge_index, edge_weight, return_attention_weights=False)
        
        x2 = x2.float()

        w=self.Aggr(x, batch)

        x = torch.sum(w*x2,dim=0)
        #x = self.fc(x)

        return x, w


class mlp(nn.Module):

    def __init__(self):
        super(mlp, self).__init__()

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Normalización

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.3)  # Regularización

        self.fc4_primary = nn.Linear(64, 4)
     

    def forward(self, x):
        #x = self.fc1(x)
        #x = self.bn1(x)
        #x = F.relu(x)

        #x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x1 = self.fc4_primary(x)
       

        return x1 #x1, x2

class model_graph(nn.Module):
    def __init__(self):
        super(model_graph, self).__init__()
        self.graphgat1 = gat()  
        self.mlp_out1 = mlp()
        self.graphgat2 = gat()
        self.mlp_out2 = mlp()
        self.adj1 = adj()
        self.adj2 = adj()

    def forward(self, fea_o, y = None):
        valid_nodes_mask = (fea_o.sum(-1) != 0)
        num_els = valid_nodes_mask.sum(dim=1).int()  # Guarda los valores de nº de nodos validos
        valid_node_indices = [mask.nonzero(as_tuple=True)[0].tolist() for mask in valid_nodes_mask] # Guarda las posiciones de los nodos validos

        all_outputs1= []
        all_weights1 = []

        for b, nel in enumerate(num_els.tolist()):
            if nel == 0:
                continue
            ind = valid_node_indices[b]
            x = fea_o[b, ind] #Seleccion de los nodos validos segun sus indices 

            # Usar el módulo adj para construir grafo con atención
            edge_index1, edge_attr1 = build_attention_graph(x, self.adj1, k=30)

            data_list1=data.Data(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
           
            out1, w1= self.graphgat1(data_list1)  
            w1 = w1.view(-1)
            all_weights1.append(w1)

            all_outputs1.append(out1)

        all_outputs1 = torch.stack(all_outputs1)

        x1 = self.mlp_out1(all_outputs1)
     
        return x1

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)