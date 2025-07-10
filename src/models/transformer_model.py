import math
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
import numpy as np
import os
import torch

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)
    
    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)
    
    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class NodeEdgeBlock(nn.Module):
    def __init__(self, hidden_dims, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Projection des arêtes vers l'espace global
        self.Etoy = nn.Linear(hidden_dims, hidden_dims)
        
        # Attention multi-têtes
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Couche de sortie
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU()
        )
    
    def forward(self, E, y):
        # ✅ MODIFICATION: Traiter E comme X (edge-list au lieu de matrice d'adjacence)
        # E shape: [batch, num_edges, num_edge_types] au lieu de [batch, num_nodes, num_nodes, num_edge_types]
        batch_size, num_edges, _ = E.shape
        
        # Projection des arêtes et reshape pour l'attention
        y_from_E = self.Etoy(E)  # [B, num_edges, H]
        y_from_E = y_from_E.mean(dim=1)  # [B, H] - moyenne sur les arêtes
        
        # Reshape y pour qu'il soit 3D comme y_from_E
        if y.dim() == 2:
            y = y.unsqueeze(1)  # [B, 1, H]
        
        # Calcul de l'attention
        y_attended, _ = self.attention(y, y_from_E.unsqueeze(1), y_from_E.unsqueeze(1))
        
        # Combinaison des features
        y = y + y_attended
        
        # Couche de sortie
        y = self.output_layer(y)
        
        return y

# ================= NOUVEAU: TÊTES TEMPORELLES (IDENTIQUE À LA PARTIE STATIQUE) =================
class TemporalAttentionBlock(nn.Module):
    def __init__(self, hidden_dims, num_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Projection des arêtes vers l'espace global (IDENTIQUE À NodeEdgeBlock)
        self.Etoy = nn.Linear(hidden_dims, hidden_dims)
        
        # Attention temporelle avec 4 têtes (SEULE DIFFÉRENCE: 4 têtes au lieu de 8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims,
            num_heads=num_heads,  # 4 têtes temporelles
            dropout=dropout,
            batch_first=True
        )
        
        # Couche de sortie (IDENTIQUE À NodeEdgeBlock)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU()
        )
    
    def forward(self, E, y):
        # ✅ EXACTEMENT IDENTIQUE À NodeEdgeBlock
        # E shape: [batch, num_edges, num_edge_types] au lieu de [batch, num_nodes, num_nodes, num_edge_types]
        batch_size, num_edges, _ = E.shape
        
        # Projection des arêtes et reshape pour l'attention (IDENTIQUE)
        y_from_E = self.Etoy(E)  # [B, num_edges, H]
        y_from_E = y_from_E.mean(dim=1)  # [B, H] - moyenne sur les arêtes
        
        # Reshape y pour qu'il soit 3D comme y_from_E (IDENTIQUE)
        if y.dim() == 2:
            y = y.unsqueeze(1)  # [B, 1, H]
        
        # Calcul de l'attention (IDENTIQUE À NodeEdgeBlock)
        print("            Formule PyTorch : y_attended, _ = self.attention(y, y_from_E.unsqueeze(1), y_from_E.unsqueeze(1))")
        print("            Formule math : y_{attended} = Attention(y, y_{from_E}, y_{from_E})")
        y_attended, _ = self.attention(y, y_from_E.unsqueeze(1), y_from_E.unsqueeze(1))
        
        # Combinaison des features (IDENTIQUE)
        y = y + y_attended
        
        # Couche de sortie (IDENTIQUE)
        y = self.output_layer(y)
        
        return y

class XEyTransformerLayer(nn.Module):
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, mode="static") -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.mode = mode
        
        # Attention spatiale (8 têtes) - CODE EXISTANT INCHANGÉ
        self.self_attn = NodeEdgeBlock(dy, n_head, dropout, **kw)
        
        # NOUVEAU: Attention temporelle (4 têtes) - MAINTENANT IDENTIQUE À self_attn
        self.temporal_attn = TemporalAttentionBlock(dy, num_heads=4, dropout=dropout, **kw)
        
        # Reste du code EXACTEMENT IDENTIQUE
        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)
        
        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)
        
        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)
        
        self.activation = F.relu
    
    def forward(self, X: Tensor, E: Tensor, y: Tensor):
        print(f"          🔥 XEyTransformerLayer - Début")
        
        # Attention
        if self.mode == "static":
            print(f"          🎯 Attention spatiale (8 têtes)")
            print("            Formule PyTorch : new_y = self.self_attn(E, y)")
            print("            Formule math : new_y = Attention(y, E)")
            new_y = self.self_attn(E, y)
        else:
            print(f"          ⏰ Attention temporelle (4 têtes)")
            print("            Formule PyTorch : new_y = self.temporal_attn(E, y)")
            print("            Formule math : new_y = TemporalAttention(y, E)")
            new_y = self.temporal_attn(E, y)
        
        # Normalisation + résidu sur y
        print("          🔄 Mise à jour y (attention + résidu + norm)")
        print("            PyTorch : y = self.norm_y1(y + self.dropout_y1(new_y))")
        print("            Math : y = LayerNorm(y + Dropout(new_y))")
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)
        
        # Feedforward X
        print("          🔄 Mise à jour X (feedforward + résidu + norm)")
        print("            PyTorch : X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X))))))")
        print("            Math : X = LayerNorm(X + Dropout(MLP(X)))")
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)
        
        # Feedforward E
        print("          🔄 Mise à jour E (feedforward + résidu + norm)")
        print("            PyTorch : E = self.normE2(E + self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E))))))")
        print("            Math : E = LayerNorm(E + Dropout(MLP(E)))")
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        
        # Feedforward y
        print("          🔄 Mise à jour finale y (feedforward + résidu + norm)")
        print("            PyTorch : y = self.norm_y2(y + self.dropout_y3(self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))))")
        print("            Math : y = LayerNorm(y + Dropout(MLP(y)))")
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)
        
        print(f"          ✅ XEyTransformerLayer - Terminé")
        return X, E, y

class GraphTransformer(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), mode="static"):
        super().__init__()
        
        self.n_layers = n_layers
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.mode = mode  # NOUVEAU: mode statique ou temporel
        
        # Embedding layers - CODE EXISTANT INCHANGÉ
        self.embedding_X = nn.Linear(input_dims['X'], hidden_dims['X'])
        self.embedding_E = nn.Linear(input_dims['E'], hidden_dims['E'])
        self.embedding_y = nn.Linear(input_dims['y'], hidden_dims['y'])
        
        # Transformer layers avec mode
        self.layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dims['X'],
                de=hidden_dims['E'],
                dy=hidden_dims['y'],
                n_head=8,  # 8 têtes spatiales (inchangé)
                dim_ffX=hidden_mlp_dims['X'],
                dim_ffE=hidden_mlp_dims['E'],
                dim_ffy=hidden_mlp_dims['y'],
                mode=self.mode  # NOUVEAU: passer le mode
            ) for _ in range(n_layers)
        ])
        
        # Output layers - CODE EXISTANT INCHANGÉ
        self.output_layer_X = nn.Linear(hidden_dims['X'], output_dims['X'])
        self.output_layer_E = nn.Linear(hidden_dims['E'], output_dims['E'])
        self.output_layer_y = nn.Linear(hidden_dims['y'], output_dims['y'])
        
        # Activation functions - CODE EXISTANT INCHANGÉ
        self.act_fn_in = act_fn_in
        self.act_fn_out = act_fn_out
    
    def forward(self, X, E, y, node_mask=None, edge_mask=None):
        print(f"      🔄 GraphTransformer - Début forward pass")
        
        # Apply embedding layers - CODE EXISTANT INCHANGÉ
        print(f"      🔗 Embeddings transformer...")
        print(f"         FiLM Embeddings:")
        print(f"         - X: {X.shape} -> {self.hidden_dims['X']}")
        print(f"         - E: {E.shape} -> {self.hidden_dims['E']}")
        print(f"         - y: {y.shape} -> {self.hidden_dims['y']}")
        print(f"         Formule FiLM: h = γ(y) * h + β(y)")
        
        X = self.act_fn_in(self.embedding_X(X))
        E = self.act_fn_in(self.embedding_E(E))
        y = self.act_fn_in(self.embedding_y(y))
        
        # Apply transformer layers - CODE EXISTANT INCHANGÉ
        print(f"      🏗️  Couches transformer ({self.n_layers} couches)...")
        for i, layer in enumerate(self.layers):
            print(f"        📍 Couche {i+1}/{self.n_layers}")
            print(f"           FiLM Transformation:")
            print(f"           - X shape: {X.shape}")
            print(f"           - E shape: {E.shape}")
            print(f"           - y shape: {y.shape}")
            X, E, y = layer(X, E, y)
        
        # Apply output layers - CODE EXISTANT INCHANGÉ
        print(f"      🎯 Couches de sortie...")
        print(f"         FiLM Output:")
        print(f"         - X: {X.shape} -> {self.output_dims['X']}")
        print(f"         - E: {E.shape} -> {self.output_dims['E']}")
        print(f"         - y: {y.shape} -> {self.output_dims['y']}")
        
        X = self.act_fn_out(self.output_layer_X(X))
        E = self.act_fn_out(self.output_layer_E(E))
        y = self.act_fn_out(self.output_layer_y(y))
        
        print(f"      ✅ GraphTransformer - Forward terminé")
        return X, E, y