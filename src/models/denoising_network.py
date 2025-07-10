import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_model import GraphTransformer, Xtoy
from typing import Tuple, Optional
from models.transition_matrices import DiGressTransitionMatrices
from models.loss import DiGressLoss

# NOUVELLE CLASSE HELPER POUR ADAPTER Etoy AU FORMAT EDGE LIST
class EtoyForEdgeList(nn.Module):
    """Version d'Etoy adaptée pour format edge list [batch, num_edges, features]"""
    
    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)
    
    def forward(self, E):
        """E: [batch, num_edges, d] au lieu de [batch, nodes, nodes, d]"""
        m = E.mean(dim=1)      # [batch, d]
        mi = E.min(dim=1)[0]   # [batch, d]  
        ma = E.max(dim=1)[0]   # [batch, d]
        std = E.std(dim=1)     # [batch, d]
        
        z = torch.hstack((m, mi, ma, std))  # [batch, 4*d]
        out = self.lin(z)      # [batch, dy]
        return out

class DenoisingNetwork(nn.Module):
    """
     Réseau de débruitage avec GraphTransformer
    Support pour modes static/dynamic avec dimensions adaptatives
    """
    def __init__(self, num_node_classes, num_edge_classes, hidden_dim=64, num_layers=3, mode="static"):
        super().__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.hidden_dim = hidden_dim
        self.mode = mode  #  Stocker le mode
        
        #  Dimension adaptative selon le mode
        self.feature_dim = 15 if mode == "static" else 22
        
        print(f"\n Initialisation DenoisingNetwork :")
        print(f"Mode: {mode}")
        print(f"Node classes: {num_node_classes}")
        print(f"Edge classes: {num_edge_classes}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Num layers: {num_layers}")
        
        # Embeddings pour transformer les features en hidden_dim
        self.node_embedding = nn.Linear(num_node_classes, hidden_dim)
        self.edge_embedding = nn.Linear(num_edge_classes, hidden_dim)
        self.feature_embedding = nn.Linear(self.feature_dim, hidden_dim)  # Dimension adaptative
        
        # AJOUT : Modules d'agrégation X→Y et E→Y
        self.Xtoy = Xtoy(dx=hidden_dim, dy=hidden_dim)
        self.Etoy_modified = EtoyForEdgeList(d=hidden_dim, dy=hidden_dim)
        
        # Configuration des dimensions pour le transformer
        input_dims = {
            'X': hidden_dim,  # Après embedding
            'E': hidden_dim,  # Après embedding
            'y': hidden_dim   # Après embedding
        }
        
        hidden_mlp_dims = {
            'X': hidden_dim * 2,
            'E': hidden_dim * 2,
            'y': hidden_dim * 2
        }
        
        hidden_dims = {
            'dx': hidden_dim,
            'de': hidden_dim,
            'dy': hidden_dim,
            'n_head': 4,
            'dim_ffX': hidden_dim * 4,
            'dim_ffE': hidden_dim * 2
        }
        
        output_dims = {
            'X': num_node_classes,
            'E': num_edge_classes,
            'y': hidden_dim
        }

        # TRANSFORMER avec mode
        self.transformer = GraphTransformer(
            n_layers=num_layers,
            input_dims={'X': hidden_dim, 'E': hidden_dim, 'y': hidden_dim},
            hidden_mlp_dims={'X': hidden_dim * 2, 'E': hidden_dim * 2, 'y': hidden_dim * 2},
            hidden_dims={'X': hidden_dim, 'E': hidden_dim, 'y': hidden_dim},
            output_dims={'X': num_node_classes, 'E': num_edge_classes, 'y': hidden_dim},
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            mode=mode  # Passer le mode au transformer
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.parameters())
        print(f" DenoisingNetwork initialisé avec {total_params:,} paramètres")

    def set_mode(self, new_mode):
        """
        Méthode pour changer le mode (entraînement bi-phasé)
        Gère automatiquement le changement de dimension des features
        """
        if self.mode != new_mode:
            print(f"\n Changement de mode DenoisingNetwork: {self.mode} → {new_mode}")
            self.mode = new_mode
            
            # Calculer la nouvelle dimension des features
            new_feature_dim = 15 if new_mode == "static" else 22
            
            if new_feature_dim != self.feature_dim:
                print(f"Dimension features: {self.feature_dim} → {new_feature_dim}")
                self.feature_dim = new_feature_dim
                
                # RÉINITIALISER l'embedding des features avec la nouvelle dimension
                old_embedding = self.feature_embedding
                self.feature_embedding = nn.Linear(self.feature_dim, self.hidden_dim)
                
                # Copier sur le même device si nécessaire
                if next(self.parameters()).is_cuda:
                    self.feature_embedding = self.feature_embedding.cuda()
                
                # Initialiser avec les mêmes poids si possible (pour la continuité)
                with torch.no_grad():
                    if new_feature_dim <= old_embedding.in_features:
                        # Si nouvelle dim <= ancienne, copier une partie des poids
                        self.feature_embedding.weight[:, :new_feature_dim] = old_embedding.weight[:, :new_feature_dim]
                        self.feature_embedding.bias = old_embedding.bias
                    else:
                        # Si nouvelle dim > ancienne, copier et initialiser le reste aléatoirement
                        min_dim = min(new_feature_dim, old_embedding.in_features)
                        self.feature_embedding.weight[:, :min_dim] = old_embedding.weight[:, :min_dim]
                        self.feature_embedding.bias = old_embedding.bias
                        # Le reste est déjà initialisé aléatoirement par nn.Linear
                
                del old_embedding  # Libérer la mémoire
            
            # METTRE À JOUR le mode du transformer
            self.transformer.mode = new_mode
            for layer in self.transformer.layers:
                layer.mode = new_mode
            
            print(f"Mode changé avec succès: {new_mode}")
            print(f"Features dimension: {self.feature_dim}")
            print(f"Transformer mode: {self.transformer.mode}")

    def forward(self, X, E, features):
        """
        Forward pass du réseau de débruitage
        
        Args:
            X: [batch_size, num_nodes, num_node_classes] ONE-HOT
            E: [batch_size, num_nodes, num_nodes, num_edge_classes] ONE-HOT
            features: [batch_size, feature_dim]  # 15 pour static, 23 pour dynamic
            
        Returns:
            loss: Loss totale (cross-entropy nodes + edges)
        """
        print(f" DenoisingNetwork - Début forward pass")
        
        # VÉRIFICATION: Dimension des features cohérente avec le mode
        expected_feature_dim = self.feature_dim
        actual_feature_dim = features.shape[-1]
        
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f" Dimension features incompatible: "
                f"attendu {expected_feature_dim} pour mode '{self.mode}', "
                f"reçu {actual_feature_dim}"
            )
        
        # Transformer les features en hidden_dim
        print(f" Embeddings...")
        X_embedded = self.node_embedding(X)  # [B, N, H]
        E_onehot_only = E[..., 2:]  # [B, num_edges, num_edge_classes]
        E_embedded = self.edge_embedding(E_onehot_only)  # [B, num_edges, H]
        features_embedded = self.feature_embedding(features)  # [B, H]
        
        # AJOUT : Agrégation X→Y et E→Y
        y_from_X = self.Xtoy(X_embedded)  # [B, H] - agrégation des nœuds
        y_from_E = self.Etoy_modified(E_embedded)  # [B, H] - agrégation des arêtes
        
        # COMBINAISON : Addition simple (garde la même dimension)
        y_combined = features_embedded + y_from_X + y_from_E
        
        # Passer à travers le transformer avec Y enrichi
        print(f" Passage dans le transformer...")
        X_pred, E_pred, _ = self.transformer(X_embedded, E_embedded, y_combined)
        
        # Calculer la loss cross-entropy
        print(f" Calcul des losses...")
        node_loss = F.cross_entropy(
            X_pred.view(-1, self.num_node_classes),
            X.argmax(dim=-1).view(-1)
        )
        
        edge_loss = F.cross_entropy(
            E_pred.view(-1, self.num_edge_classes),
            E_onehot_only.argmax(dim=-1).view(-1)
        )
        
        total_loss = node_loss + edge_loss
        print(f" DenoisingNetwork - Forward terminé (Loss: {total_loss.item():.4f})")
        return total_loss
    
    def predict(self, X, E, features):
        """
        Méthode de prédiction sans calcul de loss
        Utile pour la génération/sampling
        
        Returns:
            X_pred: [batch_size, num_nodes, num_node_classes] logits
            E_pred: [batch_size, num_nodes, num_nodes, num_edge_classes] logits
        """
        # Vérification de dimension
        expected_feature_dim = self.feature_dim
        actual_feature_dim = features.shape[-1]
        
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f" Dimension features incompatible: "
                f"attendu {expected_feature_dim} pour mode '{self.mode}', "
                f"reçu {actual_feature_dim}"
            )
        
        # Embeddings
        X_embedded = self.node_embedding(X)
        E_onehot_only = E[..., 2:]
        E_embedded = self.edge_embedding(E_onehot_only)
        features_embedded = self.feature_embedding(features)
        
        # Agrégation X→Y et E→Y
        y_from_X = self.Xtoy(X_embedded)
        y_from_E = self.Etoy_modified(E_embedded)
        y_combined = features_embedded + y_from_X + y_from_E
        
        # Forward pass transformer
        X_pred, E_pred, _ = self.transformer(X_embedded, E_embedded, y_combined)
        
        return X_pred, E_pred
    
    def get_num_parameters(self) -> int:
        """Compte le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """ ENRICHI: Informations complètes sur le modèle"""
        return {
            'mode': self.mode,
            'num_node_classes': self.num_node_classes,
            'num_edge_classes': self.num_edge_classes,
            'hidden_dim': self.hidden_dim,
            'feature_dim': self.feature_dim,
            'num_layers': len(self.transformer.layers),
            'num_parameters': self.get_num_parameters()
        }

    def get_feature_requirements(self) -> dict:
        """ NOUVEAU: Retourne les exigences de features selon le mode"""
        return {
            'static': {
                'feature_dim': 15,
                'description': 'Features statiques DiGress (temporel + structurel + spectral)'
            },
            'dynamic': {
                'feature_dim': 22,
                'description': 'Features temporelles (évolution + volatilité + prédictibilité)'
            },
            'current_mode': self.mode,
            'current_feature_dim': self.feature_dim
        }

    def validate_inputs(self, X, E, features):
        """ NOUVEAU: Validation complète des inputs"""
        batch_size = X.shape[0]
        
        # Vérifier les dimensions
        assert len(X.shape) == 3, f"X doit être 3D [batch, nodes, classes], reçu {X.shape}"
        assert len(E.shape) == 4, f"E doit être 4D [batch, nodes, nodes, classes], reçu {E.shape}"
        assert len(features.shape) == 2, f"Features doit être 2D [batch, features], reçu {features.shape}"
        
        # Vérifier la cohérence batch
        assert E.shape[0] == batch_size, f"Batch size incohérent E: {E.shape[0]} vs {batch_size}"
        assert features.shape[0] == batch_size, f"Batch size incohérent features: {features.shape[0]} vs {batch_size}"
        
        # Vérifier les classes
        assert X.shape[-1] == self.num_node_classes, f"Classes nœuds: {X.shape[-1]} vs {self.num_node_classes}"
        assert E.shape[-1] == self.num_edge_classes, f"Classes arêtes: {E.shape[-1]} vs {self.num_edge_classes}"
        
        # Vérifier dimension features
        assert features.shape[-1] == self.feature_dim, f"Features dim: {features.shape[-1]} vs {self.feature_dim}"
        
        # Vérifier que X et E sont en one-hot (somme = 1 par élément)
        X_sums = torch.sum(X, dim=-1)
        E_sums = torch.sum(E, dim=-1)
        
        assert torch.allclose(X_sums, torch.ones_like(X_sums), atol=1e-6), "X doit être en one-hot"
        assert torch.allclose(E_sums, torch.ones_like(E_sums), atol=1e-6), "E doit être en one-hot"
        
        return True