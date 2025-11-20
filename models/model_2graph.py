import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.nn import GATConv
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from torch import Tensor
from torch_geometric.nn import MLP
from typing import Optional
from torch_geometric.nn.aggr import Aggregation
from sklearn.decomposition import PCA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_attention_graph(node_features, adj_model, k=8, exclude_self=True, candidate_k=30):
    """
    Construye un grafo filtrado por atención usando preselección k-NN para reducir O(N^2).

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
    candidate_k : int
        Número de vecinos candidatos preseleccionados por nodo.

    Returns
    -------
    edge_index : LongTensor  (2, E)
        Índices (source, target) de las aristas seleccionadas.
    edge_attr  : FloatTensor (E,1)
        Peso continuo de cada arista, en el mismo orden que edge_index.
    """
    N, D = node_features.size()
    device = node_features.device

    max_candidates = N - 1 if exclude_self else N
    candidate_k = min(candidate_k, max_candidates)
    k = min(k, candidate_k)

    # --- 1) Distancias entre nodos (vectorizado) ---
    dist = torch.cdist(node_features, node_features).to(device)  # (N, N)
    if exclude_self:
        dist.fill_diagonal_(float('inf'))

    # --- 2) Selección de candidate_k vecinos más cercanos ---
    knn_idx = dist.topk(candidate_k, largest=False).indices  # (N, candidate_k)

    # --- 3) Construir pares candidatos ---
    row = torch.arange(N, device=device).unsqueeze(1).expand(-1, candidate_k).reshape(-1)
    col = knn_idx.reshape(-1)

    # --- 4) Evaluar adj_model SOLO en esos pares ---
    node_i, node_j = node_features[row].to(device), node_features[col].to(device)
    att_weights = adj_model(node_i, node_j).squeeze().to(device)  # (N*candidate_k,)

    # --- 5) Top-k por nodo según atención ---
    edge_index_list, edge_weight_list = [], []
    for i in range(N):
        sel = (row == i)
        att_i, col_i = att_weights[sel], col[sel]
        topk_idx = torch.topk(att_i, min(k, att_i.size(0)), largest=True).indices.to(device)

        edge_index_list.append(
            torch.stack([torch.full_like(col_i[topk_idx], i), col_i[topk_idx]], dim=0)
        )
        edge_weight_list.append(att_i[topk_idx])

    # --- 6) Concatenar resultados ---
    if len(edge_index_list) == 0:
        return torch.empty((2,0), dtype=torch.long, device=device), torch.empty((0,1), device=device)

    edge_index = torch.cat(edge_index_list, dim=1)          # (2, E)
    edge_attr  = torch.cat(edge_weight_list, dim=0).view(-1, 1)  # (E,1)

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

    def forward(self, node_i, node_j):
        pair = torch.cat([node_i, node_j], dim=-1)  # (E, 768*2)
        return torch.sigmoid(self.mlp(pair)).to(device)  # (E, 1)

class AttentionalAggregation2(Aggregation):
    r"""Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
            node-level gating) or :obj:`[1, out_channels]` (for feature-level
            gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.gate_nn = self.gate_mlp = None
        if isinstance(gate_nn, MLP):
            self.gate_mlp = gate_nn
        else:
            self.gate_nn = gate_nn

        self.nn = self.mlp = None
        if isinstance(nn, MLP):
            self.mlp = nn
        else:
            self.nn = nn

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2, batch:Optional[Tensor] = None) -> Tensor:

        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = torch.softmax(gate, dim=dim) 
        h = gate*x
        return h, gate

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'gate_nn={self.gate_mlp or self.gate_nn}, '
                f'nn={self.mlp or self.nn})')

class gat(torch.nn.Module):
    def __init__(self, heads=4, dim=1):
        super(gat,self).__init__()

        self.gat1 = GATConv(768, 256, heads=heads, edge_dim = dim, concat=False) # Prueba 16 / Prueba 17(127)/Prueba 18(32)
        self.gat2 = GATConv( 256, 64, heads=heads, edge_dim = dim, concat=False) # Prueba 16 / Prueba 17(127)/Prueba 18(32)
        att_net = nn.Sequential(nn.Linear(768, 1))
        
        weights_init(att_net)
        self.pool = AttentionalAggregation2(att_net)
        
    def forward(self,data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr,data.batch
        x,w=self.pool(x, batch)
        x, _ = self.gat1(x, edge_index, edge_weight, return_attention_weights=False)
        x, _ = self.gat2(x, edge_index, edge_weight, return_attention_weights=False)
        x = torch.sum(x,dim=0)
        return x,w
    
class mlp(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    
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

        all_outputs1, all_outputs2 = [],[]
        all_weights1, all_weights2  = [],[]

        for b, nel in enumerate(num_els.tolist()):
            if nel == 0:
                continue
            ind = valid_node_indices[b]
            x = fea_o[b, ind] #Seleccion de los nodos validos segun sus indices 

            # Usar el módulo adj para construir grafo con atención
            edge_index1, edge_attr1 = build_attention_graph(x, self.adj1, k=8)
            edge_index2, edge_attr2 = build_attention_graph(x, self.adj2, k=8)

            data_list1=data.Data(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
            data_list2=data.Data(x=x, edge_index=edge_index2, edge_attr=edge_attr2)


            out1, w1= self.graphgat1(data_list1)  
            w1 = w1.view(-1)
            all_weights1.append(w1)

            out2, w2= self.graphgat2(data_list2)  
            w2 = w2.view(-1)
            all_weights2.append(w2)

            all_outputs1.append(out1)
            all_outputs2.append(out2)

        all_outputs1 = torch.stack(all_outputs1)
        all_outputs2 = torch.stack(all_outputs2)

        x1 = self.mlp_out1(all_outputs1)
        x2 = self.mlp_out2(all_outputs2)

        return x1, x2
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)