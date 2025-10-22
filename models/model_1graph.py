import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.nn import GATConv, AttentionalAggregation, AGNNConv
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



class gat(torch.nn.Module):
    def __init__(self, heads=4):
        super(gat,self).__init__()

        self.gat1 = GATConv(768, 512, heads=heads, edge_dim = 1)
        self.gat2 = GATConv(512*heads, 64, heads=heads, edge_dim = 1)
        att_net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
                                nn.Tanh())
        self.pool = AttentionalAggregation(att_net)
       
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
        self.fc4_secondary = nn.Linear(64, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x1 = self.fc4_primary(x)
        x2 = self.fc4_secondary(x)

        return x1, x2

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
        x1, x2 = self.mlp_out(bf_x)

        return x1, x2

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)