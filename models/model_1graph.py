import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch import Tensor
from typing import Tuple
from torch_geometric.data import Data
import os
from transformers import AutoModel 
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
#from torch_scatter import scatter_max


from peft import get_peft_model,IA3Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available(), "  n_gpus:", torch.cuda.device_count())


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_keep_ia3_model():
    model_name = "Astaxanthin/KEEP"

    # 1. Cargar modelo en 8 bits o 4 bits (MUCHO AHORRO DE MEMORIA)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_8bit=True,        # <--- MUY IMPORTANTE
        _from_auto = True
    )

    # 2. Activar gradient checkpointing para reducir VRAM
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True

    # 3. Reducir uso de memoria durante forward
    model.config.use_cache = False

    # 4. Activar torch.compile si usas PyTorch 2.0+
    model = torch.compile(model)

    # 5. Identificar módulos lineales donde aplicar IA3
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["mlp.fc1", "mlp.fc2"]
        ):
            target_modules.append(name)

    ia3_config = IA3Config(
        target_modules=target_modules,
        feedforward_modules=target_modules,
        task_type="FEATURE_EXTRACTION"
    )

    ia3_model = get_peft_model(model, ia3_config)

    # 6. Congelar TODO menos IA3
    for name, param in ia3_model.named_parameters():
        #if "fc1.ia3" not in name :
        if "blocks.23.mlp.fc1.ia3" not in name:
            param.requires_grad = False
    # 7. Envolver con tu modelo final
    net = KEEPForGleason(ia3_model, model_graph)

    return net

class KEEPForGleason(nn.Module):
    def __init__(self, base_model, classifier_model):
        super().__init__()
        self.base = base_model

        # Cabezal más expresivo con normalización, activación y dropout
        self.classifier = classifier_model()
        self.classifier.apply(weights_init)

    def forward(self, images):
        # Codificación del encoder visual
        features_list = [torch.stack([self.base.encode_image(patch.unsqueeze(dim=0).to(dtype=torch.float16)) for patch in img.squeeze(0)])for img in images]
        logits = self.classifier(features_list)
        return logits

class model_graph(nn.Module):
    def __init__(self):
        super(model_graph, self).__init__()
        self.graphgat = gat()
        #self.adj = adj()

        # mover módulos al device por defecto
        self.to(device)

    def forward(self, x, y=None):
        """
        x: tensor (B, N, D) 
        """
        graphs_list = [grafo_fully_connected(xi.squeeze(dim=1)) for xi in x]
       
        out, w1 = self.graphgat(graphs_list)
        return out, w1

def grafo_fully_connected(x):
    num_nodos = x.shape[0]
    
    # Creamos todas las combinaciones posibles
    fila = torch.arange(num_nodos).repeat_interleave(num_nodos)
    col = torch.arange(num_nodos).repeat(num_nodos)
    
    # Quitamos autoconexiones si quieres
    mask = fila != col
    edge_index = torch.stack([fila[mask], col[mask]], dim=0)

    return Data(x=x, edge_index=edge_index)

class gat(torch.nn.Module):
    def __init__(self, heads=8):
        super(gat, self).__init__()
        self.gat1 = GATv2Conv(768, 256, heads=heads, concat=False)
        self.gat2 = GATv2Conv(256, 64, heads=heads, concat=False)

        self.Aggr = Aggregate_Attnt(L=768, I=1)   # Debe aceptar batch
        self.fc = mlp(in_dim=64, out_dim=4)

    def forward(self, graph_list):
        # 1) Empaquetar graphs en un solo batch

        batch = Batch.from_data_list(graph_list)
        batch = batch.to(device)

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        lengths = batch.ptr[1:] - batch.ptr[:-1]   # patches por WSI

        # 2) GATConv en TODO el batch a la vez (¡sin bucles!)
        z = self.gat1(x, edge_index)
        z = F.relu(z)

        z = self.gat2(z, edge_index)
        z = F.relu(z)

        # 3) Clasificador por patch
        logits = self.fc(z)
        probs = torch.softmax(logits, dim=1)

        # 4) MIL aggregation por WSI usando batch_idx
        w = self.Aggr(x, batch_idx)      # tu atención debe soportar batching

        # 5) Weighted max per WSI
        weighted = w * probs             # (N_total, 4)

        # 6) Reconstruir patch predictions por WSI
        patch_output = torch.split(weighted, lengths.tolist(), dim=0)
        out_wsi = [torch.max(p,dim=0)[0] for p in patch_output]

        return out_wsi, patch_output

class Aggregate_Attnt(nn.Module):
    def __init__(self, L=512, I=1):
        super().__init__()
        D = L // 2
        self.branch_1 = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.branch_2 = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.branch_3 = nn.Linear(D, I)
        weights_init(self)

    def forward(self, x: Tensor, dim: int = -2):
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa.mul(xb)
        A = self.branch_3(A)
        gate = torch.softmax(A, dim=0)
        return gate


class mlp(nn.Module):
    def __init__(self, in_dim=128, out_dim=4):
        super(mlp, self).__init__()
        # BatchNorm1d requiere input (B, C) — si tienes (N_nodes, C) está bien.
        self.bn = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, out_dim)
        #weights_init(self)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        #x = self.bn1(x)
        #x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc(x)
        return x
    
'''class adj(nn.Module):
    """
    MLP que da un score escalar entre 0 y 1 para cada par (node_i, node_j).
    He reducido dims intermedios para ahorrar memoria; si necesitas capacidad,
    aumenta hidden1/hidden2.
    """
    def __init__(self, in_dim=768, hidden1=256, hidden2=64):
        super(adj, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        self.sig = nn.Sigmoid()
        weights_init(self)

    def forward(self, node_i: Tensor, node_j: Tensor) -> Tensor:
        # node_i: (M, D), node_j: (M, D)  -> output (M,)
        pair = torch.cat([node_i, node_j], dim=-1)
        out = self.mlp(pair)
        return self.sig(out).view(-1)


def build_attention_graph_chunked(node_features: Tensor,
                                  adj_model: nn.Module,
                                  k: int = 8,
                                  exclude_self: bool = True,
                                  chunk_size: int = 1024
                                  ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Construye edge_index y edge_attr sin materializar NxN completo.
    - node_features: (N, D)
    - adj_model: recibe (M, D) y devuelve (M,) con gradientes
    - chunk_size: número de targets a procesar por chunk para reducir memoria
    Devuelve edge_index (2, E) y edge_attr (E,)
    """

    N, D = node_features.shape
    dev = node_features.device

    if N == 0:
        return torch.empty((2, 0), dtype=torch.long, device=dev), torch.empty((0,), dtype=torch.float, device=dev)

    edge_index_parts = []
    edge_attr_parts = []

    # Procesamos por cada nodo fuente i
    # Para cada i, evaluamos scores contra todos los j en chunks
    for i in range(N):
        src = node_features[i].unsqueeze(0)  # (1, D)
        scores_chunks = []
        idx_chunks = []

        # recorrer targets en chunks para evitar NxN
        j = 0
        while j < N:
            j_end = min(N, j + chunk_size)
            targets = node_features[j:j_end]  # (chunk, D)

            # si exclude_self y i dentro del chunk, lo saltamos manualmente
            if exclude_self and (j <= i < j_end):
                # concatenamos los targets sin el self
                if j == i and j_end == i + 1:
                    # chunk contiene sólo self -> saltar
                    j = j_end
                    continue
                # else: split the chunk into up to two parts skipping self
                parts = []
                if j < i:
                    parts.append((j, i))
                if i + 1 < j_end:
                    parts.append((i + 1, j_end))
                # evaluar cada parte
                for (a, b) in parts:
                    tgt = node_features[a:b]
                    src_rep = adj_model(src, tgt)  # modificar adj para soportar broadcasting
                    # (M, D)
                    # (M,)
                    scores_chunks.append(sc)
                    idx_chunks.append(torch.arange(a, b, device=dev))
            else:
                src_rep = src.expand(targets.size(0), -1)
                sc = adj_model(src_rep, targets)  # (M,)
                scores_chunks.append(sc)
                idx_chunks.append(torch.arange(j, j_end, device=dev))

            j = j_end

        if len(scores_chunks) == 0:
            continue

        # concatenar scores para el origen i (pero en memoria sólo M ~ N)
        att_i = torch.cat(scores_chunks, dim=0)  # (N' <= N-1)
        indices_i = torch.cat(idx_chunks, dim=0)  # indices de destino relativos al original

        # Seleccionar top-k
        topk = min(k, att_i.numel())
        if topk == 0:
            continue
        topk_idx = torch.topk(att_i, topk, largest=True).indices  # índices en att_i
        selected_targets = indices_i[topk_idx]                     # destinos reales

        # construir edge_index para este i
        src_tensor = torch.full_like(selected_targets, i, dtype=torch.long, device=dev)
        edge_index_parts.append(torch.stack([src_tensor, selected_targets], dim=0))
        edge_attr_parts.append(att_i[topk_idx])

    if len(edge_index_parts) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=dev), torch.empty((0,), dtype=torch.float, device=dev)

    edge_index = torch.cat(edge_index_parts, dim=1)  # (2, E)
    edge_attr = torch.cat(edge_attr_parts, dim=0)    # (E,)

    return edge_index.long(), edge_attr.float()'''

'''
# ------------------------------
# Inicializar KEEP + LoRA
# ------------------------------
def get_keep_lora_model():
    model_name = "Astaxanthin/KEEP"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]):
            target_modules.append(name)

    lora_config = LoraConfig(
        r=8,  # más capacidad representativa
        lora_alpha=16,  # proporcional al rango
        target_modules=target_modules,
        lora_dropout=0.1,  # regulariza para dataset pequeño/desbalanceado
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    lora_model = get_peft_model(model, lora_config)
    net = KEEPForGleason(lora_model, model_graph)

    # Congelar todos los parámetros que no son LoRA
    for name, param in net.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    return net'''