import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .transition_matrices import DiGressTransitionMatrices

class DigressLoss(nn.Module):
    """
    Implémentation de la fonction de perte DiGress qui combine:
    - Cross entropy pour les noeuds catégoriels
    - Cross entropy pour les arêtes catégorielles
    - Pondération par le schedule de diffusion
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, X_pred, E_pred, X, E, t):
        """
        Calcul de la perte
        
        Args:
            X_pred: Prédictions des nœuds [batch_size, num_nodes, num_node_classes]
            E_pred: Prédictions des arêtes [batch_size, num_nodes, num_nodes, num_edge_classes]
            X: Données des nœuds [batch_size, num_nodes]
            E: Données des arêtes [batch_size, num_nodes, num_nodes]
            t: Timesteps [batch_size]
            
        Returns:
            Perte totale
        """
        # Perte sur les nœuds
        X_pred = X_pred.view(-1, X_pred.size(-1))  # [batch_size * num_nodes, num_node_classes]
        X = X.view(-1)  # [batch_size * num_nodes]
        node_loss = self.ce(X_pred, X)
        
        # Perte sur les arêtes
        E_pred = E_pred.view(-1, E_pred.size(-1))  # [batch_size * num_nodes * num_nodes, num_edge_classes]
        E = E.view(-1)  # [batch_size * num_nodes * num_nodes]
        edge_loss = self.ce(E_pred, E)
        
        # Perte totale
        loss = node_loss + edge_loss
        
        return loss 

class DiGressLoss(nn.Module):
    """
    ✅ MODIFIÉ : Perte DiGress avec matrices de transition
    """
    def __init__(self, data_dir: str, num_timesteps: int = 1000, dataset_name: str = None):
        super().__init__()
        self.transition = DiGressTransitionMatrices(
            data_dir=data_dir,
            num_timesteps=num_timesteps,
            dataset_name=dataset_name
        )
        
    def forward(self, 
                node_pred: torch.Tensor, 
                edge_pred: torch.Tensor,
                node_target: torch.Tensor,
                edge_target: torch.Tensor,
                timestep: Optional[int] = None) -> torch.Tensor:
        """
        Calcule la perte DiGress
        
        Args:
            node_pred: Prédictions nœuds [batch_size, num_nodes, num_node_classes]
            edge_pred: Prédictions arêtes [batch_size, num_nodes, num_nodes, num_edge_classes]
            node_target: Cibles nœuds [batch_size, num_nodes, num_node_classes]
            edge_target: Cibles arêtes [batch_size, num_nodes, num_nodes, num_edge_classes]
            timestep: Timestep actuel (optionnel)
            
        Returns:
            Perte totale
        """
        # S'assurer que les matrices de transition sont créées
        device = node_pred.device
        self.transition._ensure_matrices_created(device)
        
        # Obtenir les matrices de transition pour le timestep actuel
        if timestep is not None:
            Qt_X = self.transition.Qt_X[timestep]  # [num_node_classes, num_node_classes]
            Qt_E = self.transition.Qt_E[timestep]  # [num_edge_classes, num_edge_classes]
        else:
            # Si pas de timestep spécifié, utiliser la dernière matrice
            Qt_X = self.transition.Qt_X[-1]
            Qt_E = self.transition.Qt_E[-1]
        
        # Appliquer les matrices de transition aux prédictions
        node_pred_transformed = torch.matmul(node_pred, Qt_X)  # [B, N, num_node_classes]
        edge_pred_transformed = torch.matmul(edge_pred, Qt_E)  # [B, N, N, num_edge_classes]
        
        # Calculer les pertes cross-entropy
        node_loss = -torch.sum(node_target * torch.log_softmax(node_pred_transformed, dim=-1))
        edge_loss = -torch.sum(edge_target * torch.log_softmax(edge_pred_transformed, dim=-1))
        
        # Normaliser par la taille du batch
        batch_size = node_pred.size(0)
        total_loss = (node_loss + edge_loss) / batch_size
        
        return total_loss 