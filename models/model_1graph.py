import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.nn import GATConv, AttentionalAggregation, AGNNConv

from torch import Tensor
from torch_geometric.nn import MLP
from typing import Optional
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

class Attention(nn.Module):
    def __init__(self, L = 512, D = 256, I = 1):
        super().__init__()
        '''Codigo de: (AtriiMIL)'''
        self.branch_1 = nn.Sequential(
            nn.Linear(L,D), 
            nn.Tanh())
        self.branch_2 = nn.Sequential(
            nn.Linear(L,D), 
            nn.Sigmoid())
        self.branch_3= nn.Linear(D,I)
        
    def forward(self, x): 
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa.mul(xb)
        A = self.branch_3(A) 
        return A
    
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
    def __init__(self, heads=4):
        super(gat,self).__init__()

        self.gat1 = GATConv(768, 512, heads=heads, edge_dim = 1, concat=False)
        self.gat2 = GATConv(512, 64, heads=heads, edge_dim = 1)
        #att_net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),nn.Tanh())
        att_net = Attention(256,128,1)
        self.pool = AttentionalAggregation2(att_net)
       
    def forward(self,data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr,data.batch
        edge_index= edge_index.to(x.device)
        x, _ = self.gat1(x, edge_index, edge_weight, return_attention_weights=False)
        x = F.relu(x)
        x, _ = self.gat2(x, edge_index, edge_weight, return_attention_weights=False)
        
        x = x.float()

        x=self.pool(x, batch)
  
        return x


class mlp(nn.Module):

    def __init__(self):
        super(mlp, self).__init__()

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Normalización

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.3)  # Regularización

        self.fc4_primary = nn.Linear(64, 4)
        #self.fc4_secondary = nn.Linear(64, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x1 = self.fc4_primary(x)
        #probs = torch.sigmoid(x1)   # ← activación sigmoide en la salida
        #x2 = self.fc4_secondary(x)

        return x1 #x1, x2

class model_gat(nn.Module):
    def __init__(self):
        super(model_gat, self).__init__()
        self.graphgat = gat()
        self.mlp_out = mlp()
        self.adj = adj()
    def forward(self, fea_o, batch_size_gnn=2):
        batch_size, max_nodes, feature_dim = fea_o.shape

        valid_nodes_mask = (fea_o.sum(-1) != 0)
        num_els = valid_nodes_mask.sum(dim=1).int()  # (B,)

        data_list = []

        for b, nel in enumerate(num_els.tolist()):
            if nel == 0:
                continue

            x = fea_o[b, :nel]  # (nel, feature_dim)

            # Usar el módulo adj para construir grafo con atención
            edge_index, edge_attr = build_attention_graph(x, self.adj, k=8)

            data_list.append(data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        if not data_list:
            raise ValueError("No valid graphs found in batch")

        all_outputs = []
        all_labels = []

        # Procesar en mini-batches de grafos
        for i in range(0, len(data_list), batch_size_gnn):
            batch_graphs = data_list[i:i + batch_size_gnn]
            g_batch = data.Batch.from_data_list(batch_graphs)
            out = self.graphgat(g_batch)  # (batch_size_gnn, hidden_dim)
            all_outputs.append(out)
            all_labels.append(g_batch.y)

        bf_x = torch.cat(all_outputs, dim=0)  # (total_graphs, hidden_dim)

        # Pasar todo por el MLP directamente
        x1 = self.mlp_out(bf_x)

        return x1

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)