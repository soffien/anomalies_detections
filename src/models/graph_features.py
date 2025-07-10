import torch
import torch.nn as nn
import numpy as np
# remplacer par des heurestiques plus robustes pour resoudre le probleme de comlexité
def compute_graph_features(graph, t, num_timesteps, device):
    """
    ✅ CORRIGÉ : Calcule y = f(Gt, t) selon DiGress Paper avec gestion d'erreurs
    
    Args:
        graph: Graphe bruité Gt avec attributs X et E
        t: Timestep actuel (1 à T)
        num_timesteps: Nombre total de timesteps T
        device: Device PyTorch
        
    Returns:
        y: Vecteur de features globales ∈ ℝᵈ pour les couches FiLM
    """
    
    # ✅ CORRECTION : Vérifications de sécurité
    if not hasattr(graph, 'X') or not hasattr(graph, 'E'):
        raise ValueError("❌ Le graphe doit avoir les attributs .X et .E")
    
    if graph.X.shape[0] == 0:
        raise ValueError("❌ Le graphe ne peut pas être vide")
    
    # ================= 1. FEATURE TEMPORELLE (OBLIGATOIRE) =================
    # Normaliser le timestep à [0, 1] comme spécifié dans le paper
    t_norm = float(t) / float(num_timesteps)  # t/T ∈ [0, 1]
    
    # ================= 2. FEATURES STRUCTURELLES GLOBALES =================
    
    # Taille du graphe
    n_nodes = graph.X.shape[0]
    
    # ✅ CORRECTION : Gestion d'erreur pour E
    try:
        adj_matrix = create_adjacency_from_E(graph.E)
    except Exception as e:
        print(f"⚠️  Erreur création matrice adjacence, utilisation matrice vide: {e}")
        adj_matrix = torch.zeros(n_nodes, n_nodes, device=device)
    
    # Nombre d'arêtes
    try:
        n_edges = count_edges(adj_matrix)
    except Exception:
        n_edges = 0
    
    # Degré moyen
    try:
        avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    except Exception:
        avg_degree = 0.0
    
    # Coefficient de clustering global
    try:
        clustering_coeff = compute_global_clustering_coefficient(adj_matrix)
    except Exception:
        clustering_coeff = 0.0
    
    # Nombre de triangles
    try:
        num_triangles = count_triangles(adj_matrix)
    except Exception:
        num_triangles = 0
    
    # Distribution des degrés (statistiques)
    try:
        degree_stats = compute_degree_statistics(adj_matrix)
    except Exception:
        degree_stats = {'mean': 0.0, 'std': 0.0, 'max': 0.0}
    
    # ================= 3. FEATURES SPECTRALES GLOBALES =================
    
    # Rayon spectral et gap spectral
    try:
        spectral_features = compute_spectral_features(adj_matrix)
    except Exception:
        spectral_features = {'radius': 0.0, 'gap': 0.0, 'trace': 0.0}
    
    # ================= 4. FEATURES ADDITIONNELLES DIGRESS =================
    
    # Densité du graphe
    try:
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0.0
    except Exception:
        density = 0.0
    
    # Densité des triangles
    try:
        max_triangles = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6 if n_nodes > 2 else 1
        triangle_density = num_triangles / max_triangles if max_triangles > 0 else 0.0
    except Exception:
        triangle_density = 0.0
    
    # Diamètre approximatif
    try:
        diameter_approx = approximate_diameter(adj_matrix)
    except Exception:
        diameter_approx = 0.0
    
    # ================= 5. CONCATENATION EN VECTEUR 1D =================
    
    features = [
        # Temporel
        t_norm,
        
        # Structurel basique
        float(n_nodes),
        float(n_edges), 
        avg_degree,
        density,
        
        # Motifs locaux
        clustering_coeff,
        float(num_triangles),
        triangle_density,
        
        # Statistiques des degrés
        degree_stats['mean'],
        degree_stats['std'],
        degree_stats['max'],
        
        # Spectral
        spectral_features['radius'],
        spectral_features['gap'],
        spectral_features['trace'],
        
        # Topologie globale
        diameter_approx
    ]
    
    # ✅ CORRECTION : Vérification que toutes les features sont valides
    safe_features = []
    for i, feat in enumerate(features):
        if np.isnan(feat) or np.isinf(feat):
            safe_features.append(0.0)
        else:
            safe_features.append(float(feat))
    
    # Créer le tenseur sur le bon device
    y = torch.tensor(safe_features, dtype=torch.float32, device=device)
    
    return y


def create_adjacency_from_E(E):
    """
    ✅ CORRIGÉ : Convertit le tenseur E en matrice d'adjacence avec gestion d'erreurs
    Support pour edge-list [num_edges, 4] et matrice d'adjacence [n, n, num_edge_types]
    """
    # ✅ NOUVEAU: Support pour edge-list format [num_edges, 4] (src_id, dst_id, one_hot_0, one_hot_1)
    if E.dim() == 2 and E.shape[1] == 4:
        # Format edge-list: [num_edges, 4] où colonnes 0,1 = src,dst et 2,3 = one_hot_edge_type
        num_edges = E.shape[0]
        
        # Extraire les IDs source et destination
        src_ids = E[:, 0].long()
        dst_ids = E[:, 1].long()
        
        # Extraire les types d'arêtes (one-hot)
        edge_types = E[:, 2:]  # [num_edges, 2]
        
        # Déterminer le nombre de nœuds (max ID + 1)
        max_node_id = max(torch.max(src_ids).item(), torch.max(dst_ids).item())
        num_nodes = max_node_id + 1
        
        # Créer la matrice d'adjacence
        adj = torch.zeros(num_nodes, num_nodes, device=E.device)
        
        # Pour chaque arête, si elle existe (type > 0), mettre 1 dans la matrice
        for i in range(num_edges):
            src, dst = src_ids[i].item(), dst_ids[i].item()
            edge_type = edge_types[i]  # [2]
            
            # Si l'arête existe (au moins un type > 0)
            if torch.any(edge_type > 0):
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0  # Graphe non-dirigé
        
        return adj
    
    # ✅ ANCIEN: Support pour matrice d'adjacence [n, n, num_edge_types]
    elif E.dim() == 3:
        n, _, num_edge_types = E.shape
        
        if num_edge_types < 2:
            raise ValueError(f"❌ E doit avoir au moins 2 types d'arêtes, reçu: {num_edge_types}")
        
        # E contient les types d'arêtes en one-hot ou probabilités
        if E.dtype in [torch.float32, torch.float64]:
            # Si E contient des probabilités, prendre l'argmax
            edge_types = torch.argmax(E, dim=-1)
        else:
            # Si E contient déjà des indices
            edge_types = E.squeeze(-1) if E.shape[-1] == 1 else torch.argmax(E, dim=-1)
        
        # Type 0 = pas d'arête, autres types = arêtes
        adj = (edge_types > 0).float()
        
        # ✅ CORRECTION : S'assurer que la matrice est symétrique pour graphes non-dirigés
        adj = torch.maximum(adj, adj.T)
        
        # Supprimer les auto-boucles
        adj.fill_diagonal_(0)
        
        return adj
    
    else:
        raise ValueError(f"❌ Format E non supporté. Attendu: [num_edges, 4] ou [n, n, num_edge_types], reçu shape: {E.shape}")


def count_edges(adj_matrix):
    """✅ CORRIGÉ : Compte le nombre d'arêtes avec gestion d'erreurs"""
    if adj_matrix.numel() == 0:
        return 0
    
    try:
        # Pour graphes non-dirigés, compter la moitié des connexions
        return int(torch.sum(adj_matrix).item() / 2)
    except Exception:
        return 0


def compute_global_clustering_coefficient(adj_matrix):
    """
    ✅ CORRIGÉ : Calcule le coefficient de clustering avec gestion d'erreurs
    """
    n = adj_matrix.shape[0]
    if n < 3:
        return 0.0
    
    try:
        # ✅ CORRECTION : Vérifier si la matrice a des connexions
        if torch.sum(adj_matrix) == 0:
            return 0.0
        
        # Convertir en numpy pour calculs efficaces
        A = adj_matrix.cpu().numpy().astype(np.float32)
        
        # ✅ CORRECTION : Gestion des matrices sparse
        if np.sum(A) == 0:
            return 0.0
        
        # Compter les triangles : trace(A³) / 6
        A2 = np.dot(A, A)
        A3 = np.dot(A2, A)
        triangles = np.trace(A3) / 6
        
        # Compter les triplets connectés
        degrees = np.sum(A, axis=1)
        triplets = np.sum(degrees * (degrees - 1)) / 2
        
        if triplets > 0 and triangles >= 0:
            return float(min(1.0, 3 * triangles / triplets))  # ✅ Limiter à 1
        else:
            return 0.0
            
    except Exception:
        return 0.0


def count_triangles(adj_matrix):
    """✅ CORRIGÉ : Compte le nombre de triangles avec optimisations"""
    try:
        n = adj_matrix.shape[0]
        if n < 3:
            return 0
        
        # ✅ CORRECTION : Vérifier si la matrice a des connexions
        if torch.sum(adj_matrix) == 0:
            return 0
        
        A = adj_matrix.cpu()
        
        # ✅ CORRECTION : Méthode plus efficace pour petites matrices
        if n <= 50:
            A3 = torch.matrix_power(A, 3)
            triangles = torch.trace(A3).item() / 6
        else:
            # Pour grandes matrices, méthode approximative
            triangles = 0
            A_np = A.numpy()
            for i in range(min(n, 20)):  # ✅ Limiter pour performance
                neighbors = np.where(A_np[i] > 0)[0]
                if len(neighbors) > 1:
                    for j in range(len(neighbors)):
                        for k in range(j+1, len(neighbors)):
                            if A_np[neighbors[j], neighbors[k]] > 0:
                                triangles += 1
            triangles = triangles / 3  # Chaque triangle compté 3 fois
        
        return int(max(0, triangles))
        
    except Exception:
        return 0


def compute_degree_statistics(adj_matrix):
    """✅ CORRIGÉ : Calcule les statistiques des degrés avec sécurité"""
    try:
        if adj_matrix.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        degrees = torch.sum(adj_matrix, dim=1)  # Degrés de tous les nœuds
        
        if len(degrees) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        # ✅ CORRECTION : Gestion des valeurs NaN/Inf
        degrees_clean = degrees[torch.isfinite(degrees)]
        
        if len(degrees_clean) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        mean_deg = float(torch.mean(degrees_clean).item())
        std_deg = float(torch.std(degrees_clean).item()) if len(degrees_clean) > 1 else 0.0
        max_deg = float(torch.max(degrees_clean).item())
        
        return {
            'mean': mean_deg,
            'std': std_deg,
            'max': max_deg
        }
        
    except Exception:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0}


def compute_spectral_features(adj_matrix):
    """✅ CORRIGÉ : Calcule les features spectrales avec robustesse"""
    try:
        n = adj_matrix.shape[0]
        if n == 0 or torch.sum(adj_matrix) == 0:
            return {'radius': 0.0, 'gap': 0.0, 'trace': 0.0}
        
        # ✅ CORRECTION : Limiter la taille pour performance
        if n > 100:
            # Pour grandes matrices, utiliser sous-échantillonnage
            indices = torch.randperm(n)[:50]
            sub_adj = adj_matrix[indices][:, indices]
        else:
            sub_adj = adj_matrix
        
        # Calculer les valeurs propres
        eigenvals = torch.linalg.eigvals(sub_adj.float())
        real_eigenvals = torch.real(eigenvals)
        
        # ✅ CORRECTION : Filtrer les valeurs valides
        valid_eigenvals = real_eigenvals[torch.isfinite(real_eigenvals)]
        
        if len(valid_eigenvals) == 0:
            return {'radius': 0.0, 'gap': 0.0, 'trace': 0.0}
        
        # Trier par ordre décroissant
        sorted_eigenvals = torch.sort(valid_eigenvals, descending=True)[0]
        
        # Rayon spectral
        spectral_radius = float(torch.max(torch.abs(valid_eigenvals)).item())
        
        # Gap spectral
        if len(sorted_eigenvals) > 1:
            spectral_gap = float((sorted_eigenvals[0] - sorted_eigenvals[1]).item())
        else:
            spectral_gap = 0.0
        
        # Trace
        trace = float(torch.sum(valid_eigenvals).item())
        
        return {
            'radius': spectral_radius,
            'gap': max(0.0, spectral_gap),
            'trace': trace
        }
        
    except Exception:
        return {'radius': 0.0, 'gap': 0.0, 'trace': 0.0}


def approximate_diameter(adj_matrix):
    """
    ✅ CORRIGÉ : Approximation robuste du diamètre du graphe
    """
    try:
        n = adj_matrix.shape[0]
        if n <= 1 or torch.sum(adj_matrix) == 0:
            return 0.0
        
        # ✅ CORRECTION : Limiter pour performance
        if n > 100:
            return float(min(10, n // 10))  # Approximation grossière pour grandes matrices
        
        A = adj_matrix.cpu().numpy()
        max_distance = 0
        
        # Échantillonner quelques nœuds pour l'approximation
        sample_size = min(3, n)  # ✅ RÉDUIT pour performance
        sample_nodes = np.random.choice(n, sample_size, replace=False)
        
        for start_node in sample_nodes:
            # BFS depuis ce nœud
            distances = np.full(n, -1)
            distances[start_node] = 0
            queue = [start_node]
            max_depth = 0  # ✅ Limiter la profondeur
            
            while queue and max_depth < 10:  # ✅ Limite de profondeur
                current = queue.pop(0)
                current_dist = distances[current]
                max_depth = max(max_depth, current_dist)
                
                # Explorer les voisins
                neighbors = np.where(A[current] > 0)[0]
                for neighbor in neighbors:
                    if distances[neighbor] == -1:  # Pas encore visité
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
            
            # Mettre à jour la distance maximale
            reachable_distances = distances[distances >= 0]
            if len(reachable_distances) > 0:
                max_distance = max(max_distance, np.max(reachable_distances))
        
        return float(min(max_distance, n))  # ✅ Limiter à n
        
    except Exception:
        return 0.0


class DiGressGraphFeatures(nn.Module):
    """
    ✅ INCHANGÉ : Module PyTorch pour calculer les features DiGress
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, graph, t, num_timesteps, device):
        """Interface principale pour l'entraînement"""
        return compute_graph_features(graph, t, num_timesteps, device)
    
    def get_feature_dim(self):
        """Retourne la dimension du vecteur de features"""
        return 15