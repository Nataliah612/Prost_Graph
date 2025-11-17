import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.nn import GATConv

from torch import Tensor
from typing import Tuple, Optional

import os
from transformers import AutoModel 

from peft import LoraConfig, get_peft_model,IA3Config
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available(), "  n_gpus:", torch.cuda.device_count())


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class adj(nn.Module):
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
                    src_rep = src.expand(tgt.size(0), -1)  # (M, D)
                    sc = adj_model(src_rep, tgt)  # (M,)
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

    return edge_index.long(), edge_attr.float()


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
    def __init__(self, in_dim=64, out_dim=4):
        super(mlp, self).__init__()
        # BatchNorm1d requiere input (B, C) — si tienes (N_nodes, C) está bien.
        self.bn = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_dim, out_dim)
        weights_init(self)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class gat(torch.nn.Module):
    def __init__(self, heads=8):
        super(gat, self).__init__()
        # GATConv( in_channels, out_channels_per_head, heads, edge_dim, concat=False )
        self.gat1 = GATConv(768, 64, heads=heads, edge_dim=1, concat=False)
        self.Aggr = Aggregate_Attnt(L=768, I=1)
        self.fc = mlp(in_dim=64, out_dim=4)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, None
        edge_index = edge_index.to(x.device)

        # GATConv puede requerir float32 si usas BatchNorm después; mantén dtype coherente.
        z,_ = self.gat1(x, edge_index, edge_weight, return_attention_weights=False)
        z = F.relu(z)

        w = self.Aggr(x)  # ojo: en tu versión original pasabas batch; aquí usamos x
        z = self.fc(z)
        z = torch.softmax(z, dim=0)
        x, x_ind = torch.max(w * z, dim=0)
        return x, w


class model_graph(nn.Module):
    def __init__(self):
        super(model_graph, self).__init__()
        self.graphgat = gat()
        self.adj = adj()

        # mover módulos al device por defecto
        self.to(device)

    def forward(self, fea_o, y=None):
        """
        fea_o: tensor (N, D) o (B, N, D) según tu pipeline original.
        En el código original había operaciones confusas de indexado; aquí
        asumo fea_o es (N, D) con nodos que pueden ser todo ceros (invalidos).
        """
        # Mascara de nodos válidos (suma != 0)
        valid_nodes_mask = (fea_o.sum(-1) != 0)
        valid_idx = torch.nonzero(valid_nodes_mask, as_tuple=False).view(-1)

        if valid_idx.numel() == 0:
            # nada válido
            return torch.empty((0, 4), device=fea_o.device), torch.empty((0,), device=fea_o.device)

        x = fea_o[valid_idx]  # (N_valid, D)

        # Construir grafo con atención con chunking para ahorrar memoria
        edge_index1, edge_attr1 = build_attention_graph_chunked(x, self.adj, k=8, chunk_size=512)

        data_list1 = data.Data(x=x, edge_index=edge_index1, edge_attr=edge_attr1)

        out1, w1 = self.graphgat(data_list1)
        w1 = w1.view(-1)

        return out1, w1


class KEEPForGleason(nn.Module):
    def __init__(self, base_model, classifier_model):
        super().__init__()
        self.base = base_model

        # Extraemos la dimensión de salida del encoder automáticamente
        rep_dim = getattr(base_model, "embed_dim", 768)

        # Cabezal más expresivo con normalización, activación y dropout
        self.classifier = classifier_model()

        # Inicialización de pesos (mejor estabilidad)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        # Codificación del encoder visual
        
        features_list = [[self.base.encode_image(patch.unsqueeze(dim=0), proj_contrast=False, normalize=False) for patch in img]for img in images]

        logits = self.classifier(features_list)
        return logits

# ------------------------------
# Inicializar KEEP + LoRA
# ------------------------------
def get_keep_ia3_model():
    model_name = "Astaxanthin/KEEP"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # -----------------------------
    # 1. Seleccionamos los módulos objetivo igual que en tu código
    # -----------------------------
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["attn.qkv", "attn.proj"]#, "mlp.fc1", "mlp.fc2"]
        ):
            target_modules.append(name)

    # -----------------------------
    # 2. Configuración IA³
    # -----------------------------
    ia3_config = IA3Config(
        target_modules=target_modules,
        feedforward_modules=target_modules, 
        # Opcional: puedes agregar más si tu modelo tiene otros nombres
        task_type="FEATURE_EXTRACTION"
    )

    # -----------------------------
    # 3. Aplicamos IA³ al modelo base
    # -----------------------------
    ia3_model = get_peft_model(model, ia3_config)

    # Wrapper final (igual que en tu código)
    net = KEEPForGleason(ia3_model, model_graph)

    # -----------------------------
    # 4. Congelar todo excepto IA³
    # -----------------------------
    for name, param in net.named_parameters():
        if "ia3_" not in name:       # IA³ usa este prefijo
            param.requires_grad = False

    return net
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