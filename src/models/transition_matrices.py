#/home/sofien/Desktop/New Folder 1/src/models/transition_matrices.py

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import os
import glob
from models.visualize_snapshot import get_snapshot_files, SnapshotLoader

class NoisyGraph:
    """Classe simple pour encapsuler les données de graphe bruitées"""
    def __init__(self, X_onehot, E_onehot):
        self.X_onehot = X_onehot
        self.E_onehot = E_onehot

def calculate_alpha_cosine(num_timesteps, s=0.008):
    """Calcule alpha selon schedule cosinus"""
    steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
    f_t = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    f_0 = f_t[0]
    alphas = f_t[1:] / f_0
    return alphas

class DiGressTransitionMatrices(nn.Module):
    """
    ✅ CORRIGÉ RADICALEMENT : Matrices de transition DiGress cohérentes
    """
    def __init__(self, num_timesteps=1000, dataset_name='bitcoinalpha'):
        super().__init__()
        
        print(f"=== INITIALISATION DIGRESS TRANSITION MATRICES ===")
        print(f"Timesteps: {num_timesteps}")
        print(f"Dataset: {dataset_name}")
        
        self.num_timesteps = num_timesteps
        
        # Obtenir le nombre total de catégories depuis le premier snapshot
        snapshot_files = get_snapshot_files(dataset_name)
        if not snapshot_files:
            raise ValueError(f"❌ Aucun snapshot trouvé pour le dataset '{dataset_name}'")
            
        # Utiliser SnapshotLoader pour obtenir le nombre de catégories
        self.num_X_categories = SnapshotLoader.get_total_categories()
        self.num_E_categories = 2  # 0 pour pas d'arête, 1 pour arête présente
        
        print(f"Nombre total de catégories X: {self.num_X_categories}")
        print(f"Nombre total de catégories E: {self.num_E_categories}")
        
        # Calcul des marginales à partir des snapshots
        self.marginal_X, self.marginal_E = self._compute_marginals_from_snapshots(
            snapshot_files, dataset_name
        )
        
        # Enregistrer comme paramètres
        self.register_parameter('marginal_dist_X', nn.Parameter(self.marginal_X, requires_grad=False))
        self.register_parameter('marginal_dist_E', nn.Parameter(self.marginal_E, requires_grad=False))
        
        # ✅ CORRECTION : Schedule de diffusion
        self.alphas = nn.Parameter(calculate_alpha_cosine(num_timesteps), requires_grad=False)
        self.betas = nn.Parameter(1 - self.alphas, requires_grad=False)
        # ✅ CORRECTION : Ajouter les matrices Q_bar_X et Q_bar_E
        self.Q_bar_X = nn.Parameter(torch.eye(self.num_X_categories), requires_grad=False)
        self.Q_bar_E = nn.Parameter(torch.eye(self.num_E_categories), requires_grad=False)
        
        print(f"✅ Configuration finale:")
        print(f"- X catégories: {self.num_X_categories}")
        print(f"- E catégories: {self.num_E_categories}")
        print(f"- Timesteps: {self.num_timesteps}")
        print(f"- Alpha range: [{self.alphas.min():.4f}, {self.alphas.max():.4f}]")
        
        self._matrices_created = False
        print(f"=== PRÊT ===\n")
        
    def _ensure_matrices_created(self, device):
        """S'assure que les matrices sont créées"""
        if not self._matrices_created:
            print(f"C Création des matrices DiGress sur device: {device}")
            
            # Créer les matrices de transition
            Qt_X_list = self._create_transition_matrices_X(device)
            Qt_E_list = self._create_transition_matrices_E(device)
            
            # Convertir les listes en tenseurs
            Qt_X = torch.stack(Qt_X_list)
            Qt_E = torch.stack(Qt_E_list)
            
            # Créer les paramètres
            self.Qt_X = nn.Parameter(Qt_X, requires_grad=False)
            self.Qt_E = nn.Parameter(Qt_E, requires_grad=False)
            
            # Calculer les matrices cumulatives
            Q_bar_X_list = self._compute_cumulative_matrices_X(device)
            Q_bar_E_list = self._compute_cumulative_matrices_E(device)
            
            # Convertir les listes en tenseurs
            Q_bar_X = torch.stack(Q_bar_X_list)
            Q_bar_E = torch.stack(Q_bar_E_list)
            
            # Créer les paramètres
            self.Q_bar_X = nn.Parameter(Q_bar_X, requires_grad=False)
            self.Q_bar_E = nn.Parameter(Q_bar_E, requires_grad=False)
            
            self._matrices_created = True
            print("✅ Matrices DiGress créées avec succès")
    
    def to(self, device):
        """Override pour gérer le device correctement"""
        super().to(device)
        
        if self._matrices_created:
            self._matrices_created = False
        
        return self
    
    def _create_transition_matrices_X(self, device):
        """Crée les matrices de transition Qt pour les nœuds"""
        Qt_X = []  # Liste de matrices 2D
        
        for t in range(self.num_timesteps):
            # Qt = betas[t] * I + alphas[t] * marginal_X
            Qt = self.betas[t] * torch.eye(self.num_X_categories, device=device) + \
                     self.alphas[t] * self.marginal_X.to(device).unsqueeze(0).expand(self.num_X_categories, -1)
            
            # Normaliser pour avoir une matrice de transition valide
            Qt = Qt / Qt.sum(dim=1, keepdim=True)
            Qt_X.append(Qt)
        
        return Qt_X

    def _create_transition_matrices_E(self, device):
        """Crée les matrices de transition Qt pour les arêtes"""
        Qt_E = []  # Liste de matrices 2D
        
        for t in range(self.num_timesteps):
            # Qt = betas[t] * I + alphas[t] * marginal_E
            Qt = self.betas[t] * torch.eye(self.num_E_categories, device=device) + \
                     self.alphas[t] * self.marginal_E.to(device).unsqueeze(0).expand(self.num_E_categories, -1)
            
            # Normaliser pour avoir une matrice de transition valide
            Qt = Qt / Qt.sum(dim=1, keepdim=True)
            Qt_E.append(Qt)
        
        return Qt_E
    
    def _compute_cumulative_matrices_X(self, device):
        """Calcule Q̄t = Q1 × Q2 × ... × Qt"""
        Q_bar_X_all = []
        current_Q_bar = torch.eye(self.num_X_categories, device=device)
        
        for t in range(self.num_timesteps):
            Qt = self.Qt_X[t].to(device)  # Qt est maintenant une matrice 2D
            # Q̄t = Q̄t-1 × Qt
            current_Q_bar = torch.matmul(current_Q_bar, Qt)
            # Normaliser pour avoir une matrice de transition valide
            current_Q_bar = current_Q_bar / current_Q_bar.sum(dim=1, keepdim=True)
            Q_bar_X_all.append(current_Q_bar.clone())
        
        return Q_bar_X_all  # Retourne une liste de matrices 2D
    
    def _compute_cumulative_matrices_E(self, device):
        """Calcule Q̄t = Q1 × Q2 × ... × Qt"""
        Q_bar_E_all = []
        current_Q_bar = torch.eye(self.num_E_categories, device=device)
        
        for t in range(self.num_timesteps):
            Qt = self.Qt_E[t].to(device)  # Qt est maintenant une matrice 2D
            # Q̄t = Q̄t-1 × Qt
            current_Q_bar = torch.matmul(current_Q_bar, Qt)
            # Normaliser pour avoir une matrice de transition valide
            current_Q_bar = current_Q_bar / current_Q_bar.sum(dim=1, keepdim=True)
            Q_bar_E_all.append(current_Q_bar.clone())
        
        return Q_bar_E_all  # Retourne une liste de matrices 2D
    
    def forward(self, device):
        """Assure que les matrices sont créées"""
        self._ensure_matrices_created(device)
        return self
    
    def get_Q_bar_t(self, t, device):
        """Obtient Q̄t pour le bruitage"""
        self._ensure_matrices_created(device)
        # ✅ CORRECTION : Utiliser t-1 pour l'indexation
        return self.Q_bar_X[t-1], self.Q_bar_E[t-1]
    
    def get_Qt(self, t, device):
        """Obtient Qt pour le timestep t"""
        self._ensure_matrices_created(device)
        return self.Qt_X[t], self.Qt_E[t]
    
    def get_marginal_distributions(self):
        """Retourne les vraies distributions marginales"""
        return self.marginal_dist_X, self.marginal_dist_E
    
    def get_categories_info(self):
        """✅ NOUVEAU : Retourne les informations sur les catégories"""
        return {
            'num_X_categories': self.num_X_categories,
            'num_E_categories': self.num_E_categories,
        }
    
    def _convert_probabilities_to_categories(self, probabilities):
        """Convertit les probabilités en catégories discrètes en suivant les probabilités cumulées"""
        # Calculer les probabilités cumulées
        cum_probs = torch.cumsum(probabilities, dim=-1)
        
        # Générer des nombres aléatoires uniformes
        random_values = torch.rand(probabilities.shape[:-1], device=probabilities.device)
        
        # Trouver la première catégorie dont la probabilité cumulée dépasse la valeur aléatoire
        random_values = random_values.unsqueeze(-1).expand_as(cum_probs)
        categories = torch.argmax((cum_probs > random_values).long(), dim=-1)
        
        return categories
    
    def apply_noise_to_graph(self, graph, t: int, device: torch.device, max_nodes: int):
        """Applique le bruit au graphe selon le timestep t"""
        print(f"\n=== 🌪️ Application du bruit au graphe (t={t}) ===")
        print(f"📊 Statistiques avant bruitage:")
        print(f"  - X: shape={graph.X.shape}")
        print(f"  - E: shape={graph.E.shape}")
        
        # One-hot encoding
        print(f"\n🔄 Conversion en one-hot...")
        X_onehot = self._convert_to_onehot_X(graph.X, max_nodes, device)
        E_onehot = self._convert_to_onehot_E(graph.E, max_nodes, device)
        
        print(f"  - X one-hot: {X_onehot.shape}")
        print(f"  - E one-hot: {E_onehot.shape}")
        
        # S'assurer que les matrices sont créées
        self._ensure_matrices_created(device)
        
        # Obtenir les matrices de transition pour ce timestep
        print(f"\n📉 Matrices de transition pour t={t}:")
        Q_bar_t_X, Q_bar_t_E = self.get_Q_bar_t(t, device)
        print(f"  - Q_bar_t_X: {Q_bar_t_X.shape}")
        print(f"  - Q_bar_t_E: {Q_bar_t_E.shape}")
        
        # ✅ OPTIMISATION: Utiliser torch.multinomial pour échantillonner efficacement
        print(f"\n🎲 Échantillonnage du bruit...")
        
        # Pour les nœuds (X)
        print(f"  Nœuds (X):")
        X_probs = torch.matmul(X_onehot, Q_bar_t_X)  # [num_nodes, num_categories]
        print(f"    - Probabilités: min={X_probs.min():.4f}, max={X_probs.max():.4f}")
        X_samples = torch.multinomial(X_probs, num_samples=1)  # [num_nodes, 1]
        print(f"    - Échantillons: shape={X_samples.shape}, valeurs uniques={torch.unique(X_samples).tolist()}")
        X_noisy_onehot = torch.zeros_like(X_onehot)
        X_noisy_onehot.scatter_(1, X_samples, 1.0)
        
        # Pour les arêtes (E)
        print(f"  Arêtes (E):")
        E_probs = torch.matmul(E_onehot[:, 2:], Q_bar_t_E)  # [num_edges, num_categories]
        print(f"    - Probabilités: min={E_probs.min():.4f}, max={E_probs.max():.4f}")
        E_samples = torch.multinomial(E_probs, num_samples=1)  # [num_edges, 1]
        print(f"    - Échantillons: shape={E_samples.shape}, valeurs uniques={torch.unique(E_samples).tolist()}")
        E_noisy_onehot = torch.zeros_like(E_onehot)
        E_noisy_onehot.scatter_(1, E_samples, 1.0)
        

        
        # ✅ Retourner directement le format one-hot
        return NoisyGraph(X_noisy_onehot, E_noisy_onehot)
    
    def _convert_to_onehot_X(self, X_raw, max_nodes, device):
        """Convertit les données X brutes en one-hot avec catégories globales
        X_raw: [num_nodes, 2] (col 0: node_id, col 1: node_category)
        """
        X_onehot = torch.zeros(max_nodes, self.num_X_categories, device=device)
        num_nodes = min(X_raw.shape[0], max_nodes)
        for i in range(num_nodes):
            category = int(X_raw[i, 1].item())
            if 0 <= category < self.num_X_categories:
                X_onehot[i, category] = 1.0
        return X_onehot
    
    def _convert_to_onehot_E(self, E_raw, max_nodes, device):
        """Convertit les données E brutes en one-hot avec catégories globales
        E_raw: [num_edges, 3] (col 0: src_id, col 1: dst_id, col 2: edge_type)
        Retourne: [num_edges, 2 + num_edge_classes] (src_id, dst_id, one_hot_edge_type)
        """
        num_edges = E_raw.shape[0]
        
        # Garder les colonnes source_id et destination_id
        src_ids = E_raw[:, 0]  # [num_edges]
        dst_ids = E_raw[:, 1]  # [num_edges]
        
        # Faire le one-hot sur la 3ème colonne (types d'arêtes)
        edge_types = E_raw[:, 2].long()  # [num_edges]
        edge_onehot = F.one_hot(edge_types, num_classes=self.num_E_categories).float().to(device)  # [num_edges, num_edge_classes]
        
        # Concaténer: [src_ids, dst_ids, edge_onehot]
        E_onehot = torch.cat([
            src_ids.unsqueeze(1),  # [num_edges, 1]
            dst_ids.unsqueeze(1),  # [num_edges, 1]
            edge_onehot  # [num_edges, num_edge_classes]
        ], dim=1)  # [num_edges, 2 + num_edge_classes]
        
        return E_onehot
    
    def _compute_marginals_from_snapshots(self, snapshot_files, dataset_name):
        """Calcule les vraies distributions marginales à partir de tous les snapshots"""
        print(f"🔍 Calcul des distributions marginales pour le dataset '{dataset_name}'...")
        
        # Utiliser SnapshotLoader pour obtenir le nombre total de catégories
        total_categories = SnapshotLoader.get_total_categories()
        
        # Initialiser les compteurs
        X_counts = torch.zeros(total_categories)
        E_counts = torch.zeros(self.num_E_categories)
        total_snapshots = 0
        
        # Traiter les snapshots par lots de 10
        batch_size = 10
        for i in range(0, len(snapshot_files), batch_size):
            batch_files = snapshot_files[i:i + batch_size]
            
            for snapshot_file in batch_files:
                # Utiliser SnapshotLoader pour charger seulement les données nécessaires
                X = SnapshotLoader.get_X(snapshot_file)
                E = SnapshotLoader.get_E(snapshot_file)
                
                # Compter les catégories de nœuds
                node_categories = X[:, 1]  # [num_nodes]
                for cat in range(self.num_X_categories):
                    X_counts[cat] += (node_categories == cat).sum()
                
                # Compter les types d'arêtes
                edge_types = E[:, 2]  # [num_edges] - prendre la 3ème colonne (types d'arêtes)
                for cat in range(self.num_E_categories):
                    E_counts[cat] += (edge_types == cat).sum()
                
                total_snapshots += 1
                
                # Libérer la mémoire
                del X
                del E
                del node_categories
                del edge_types
            
            print(f"  Traité {total_snapshots}/{len(snapshot_files)} snapshots")
        
        # Normaliser pour obtenir les distributions
        marginal_X = X_counts / X_counts.sum()
        marginal_E = E_counts / E_counts.sum()
        
        print(f"✅ Distributions marginales calculées sur {total_snapshots} snapshots")
        print(f"Distribution X: {marginal_X.tolist()}")
        print(f"Distribution E: {marginal_E.tolist()}")
        
        return marginal_X, marginal_E
    
    def _apply_categorical_noise(self, data, transition_matrix, device):
        """Applique le bruit catégorique aux données"""
        print("\n=== Application du bruit catégorique ===")
        print(f"Forme des données: {data.shape}")
        print(f"Forme de la matrice de transition: {transition_matrix.shape}")
        
        # Reshape pour la multiplication matricielle
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        
        # Simple multiplication matricielle
        noisy_data = torch.matmul(data_flat, transition_matrix)
        
        # Reshape vers la forme originale
        noisy_data = noisy_data.reshape(original_shape)
        
        print(f"Forme après bruit: {noisy_data.shape}")
        print(f"Valeurs min/max: {noisy_data.min():.4f}/{noisy_data.max():.4f}")
        
        return noisy_data