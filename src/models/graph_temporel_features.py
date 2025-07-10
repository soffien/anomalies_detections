import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import random

def compute_temporal_graph_features(graph_sequence, current_t, num_timesteps, device, window_size=5):
    """
    ✅ Calcule les features temporelles pour graphes dynamiques y = f(G_t, G_{t-1}, ..., t)
    
    Args:
        graph_sequence: Liste des graphes [G_{t-w}, ..., G_{t-1}, G_t] avec attributs X et E
        current_t: Timestep actuel (1 à T)
        num_timesteps: Nombre total de timesteps T
        device: Device PyTorch
        window_size: Taille de la fenêtre temporelle pour l'historique
        
    Returns:
        y: Vecteur de features temporelles ∈ ℝᵈ pour les couches FiLM
    """
    
    # ✅ Vérifications de sécurité
    if not graph_sequence or len(graph_sequence) == 0:
        raise ValueError("❌ La séquence de graphes ne peut pas être vide")
    
    current_graph = graph_sequence[-1]  # Graphe actuel G_t
    
    if not hasattr(current_graph, 'X') or not hasattr(current_graph, 'E'):
        raise ValueError("❌ Le graphe doit avoir les attributs .X et .E")
    
    if current_graph.X.shape[0] == 0:
        raise ValueError("❌ Le graphe ne peut pas être vide")
    
    # ================= 1. FEATURE TEMPORELLE BASIQUE =================
    # Normaliser le timestep à [0, 1]
    t_norm = float(current_t) / float(num_timesteps)
    
    # Position relative dans la séquence
    seq_position = float(len(graph_sequence)) / float(window_size)
    
    # ================= 2. ÉVOLUTION DES CONNEXIONS =================
    
    # Calculer les matrices d'adjacence pour la séquence
    adj_sequence = []
    for graph in graph_sequence:
        try:
            adj = create_adjacency_from_E(graph.E)
            adj_sequence.append(adj)
        except Exception:
            n_nodes = graph.X.shape[0]
            adj_sequence.append(torch.zeros(n_nodes, n_nodes, device=device))
    
    current_adj = adj_sequence[-1]
    n_nodes = current_adj.shape[0]
    
    # Taux d'apparition d'arêtes
    edge_birth_rate = compute_edge_birth_rate(adj_sequence)
    
    # Taux de disparition d'arêtes
    edge_death_rate = compute_edge_death_rate(adj_sequence)
    
    # Stabilité des connexions
    edge_stability = compute_edge_stability(adj_sequence)
    
    # Intermittence des connexions (optimisée)
    edge_intermittency = compute_edge_intermittency_fast(adj_sequence)
    
    # ================= 3. ÉVOLUTION STRUCTURELLE =================
    
    # Évolution du degré moyen
    degree_evolution = compute_degree_evolution(adj_sequence)
    
    # Évolution de la densité
    density_evolution = compute_density_evolution(adj_sequence)
    
    # Stabilité du voisinage (échantillonnée)
    neighborhood_stability = compute_neighborhood_stability_fast(adj_sequence)
    
    # Évolution du clustering (heuristique rapide)
    clustering_evolution = compute_clustering_evolution_fast(adj_sequence)
    
    # ================= 4. MOTIFS TEMPORELS =================
    
    # Évolution des triangles (approximation rapide)
    triangle_evolution = compute_triangle_evolution_fast(adj_sequence)
    
    # Auto-corrélation structurelle
    structural_autocorr = compute_structural_autocorrelation(adj_sequence)
    
    # Périodicité détectée
    structural_periodicity = compute_structural_periodicity(adj_sequence)
    
    # ================= 5. CENTRALITÉ TEMPORELLE =================
    
    # Évolution de la centralité de degré
    centrality_evolution = compute_centrality_evolution(adj_sequence)
    
    # Persistance des nœuds centraux
    centrality_persistence = compute_centrality_persistence(adj_sequence)
    
    # ================= 6. PRÉDICTIBILITÉ =================
    
    # Entropie de l'évolution structurelle
    structural_entropy = compute_structural_entropy(adj_sequence)
    
    # Surprise structurelle
    structural_surprise = compute_structural_surprise(adj_sequence)
    
    # Prédictibilité de formation d'arêtes (échantillonnée)
    edge_predictability = compute_edge_predictability_fast(adj_sequence)
    
    # ================= 7. MÉTRIQUES GLOBALES TEMPORELLES =================
    
    # Volatilité globale du graphe
    global_volatility = compute_global_volatility(adj_sequence)
    
    # Tendance temporelle (croissance/décroissance)
    temporal_trend = compute_temporal_trend(adj_sequence)
    
    # ================= 8. CONCATENATION EN VECTEUR 1D =================
    
    features = [
        # Temporel basique
        t_norm,
        seq_position,
        
        # Évolution des connexions
        edge_birth_rate,
        edge_death_rate,
        edge_stability,
        edge_intermittency,
        
        # Évolution structurelle
        degree_evolution['rate'],
        degree_evolution['volatility'],
        density_evolution,
        neighborhood_stability,
        clustering_evolution,
        
        # Motifs temporels
        triangle_evolution,
        structural_autocorr,
        structural_periodicity,
        
        # Centralité temporelle
        centrality_evolution['mean'],
        centrality_evolution['variance'],
        centrality_persistence,
        
        # Prédictibilité
        structural_entropy,
        structural_surprise,
        edge_predictability,
        
        # Métriques globales
        global_volatility,
        temporal_trend
    ]
    
    # ✅ Vérification que toutes les features sont valides
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
    """Convertit le tenseur E en matrice d'adjacence"""
    if E.dim() != 3:
        raise ValueError(f"❌ E doit être 3D [n, n, num_edge_types], reçu shape: {E.shape}")
    
    n, _, num_edge_types = E.shape
    
    if num_edge_types < 2:
        raise ValueError(f"❌ E doit avoir au moins 2 types d'arêtes, reçu: {num_edge_types}")
    
    if E.dtype in [torch.float32, torch.float64]:
        edge_types = torch.argmax(E, dim=-1)
    else:
        edge_types = E.squeeze(-1) if E.shape[-1] == 1 else torch.argmax(E, dim=-1)
    
    adj = (edge_types > 0).float()
    adj = torch.maximum(adj, adj.T)
    adj.fill_diagonal_(0)
    
    return adj

def compute_edge_birth_rate(adj_sequence):
    """Calcule le taux d'apparition d'arêtes"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        total_births = 0
        total_comparisons = 0
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Nouvelles arêtes = présentes maintenant mais pas avant
            new_edges = ((curr_adj > 0) & (prev_adj == 0))
            births = torch.sum(new_edges).item() / 2  # Diviser par 2 pour graphes non-dirigés
            
            # Normaliser par le nombre d'arêtes possibles
            n = curr_adj.shape[0]
            possible_edges = n * (n - 1) / 2 if n > 1 else 1
            
            total_births += births / possible_edges
            total_comparisons += 1
        
        return total_births / total_comparisons if total_comparisons > 0 else 0.0
        
    except Exception:
        return 0.0

def compute_edge_death_rate(adj_sequence):
    """Calcule le taux de disparition d'arêtes"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        total_deaths = 0
        total_comparisons = 0
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Arêtes disparues = présentes avant mais plus maintenant
            lost_edges = ((prev_adj > 0) & (curr_adj == 0))
            deaths = torch.sum(lost_edges).item() / 2
            
            # Normaliser par le nombre d'arêtes présentes avant
            prev_edge_count = torch.sum(prev_adj).item() / 2
            if prev_edge_count > 0:
                death_rate = deaths / prev_edge_count
            else:
                death_rate = 0.0
            
            total_deaths += death_rate
            total_comparisons += 1
        
        return total_deaths / total_comparisons if total_comparisons > 0 else 0.0
        
    except Exception:
        return 0.0

def compute_edge_stability(adj_sequence):
    """Calcule la stabilité des arêtes (proportion d'arêtes qui persistent)"""
    try:
        if len(adj_sequence) < 2:
            return 1.0
        
        total_stability = 0
        total_comparisons = 0
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Arêtes stables = présentes dans les deux graphes
            stable_edges = ((prev_adj > 0) & (curr_adj > 0))
            stability_count = torch.sum(stable_edges).item() / 2
            
            # Union des arêtes des deux graphes
            union_edges = ((prev_adj > 0) | (curr_adj > 0))
            union_count = torch.sum(union_edges).item() / 2
            
            if union_count > 0:
                stability = stability_count / union_count
            else:
                stability = 1.0  # Si pas d'arêtes, stabilité parfaite
            
            total_stability += stability
            total_comparisons += 1
        
        return total_stability / total_comparisons if total_comparisons > 0 else 1.0
        
    except Exception:
        return 0.0

def compute_edge_intermittency_fast(adj_sequence):
    """🚀 Calcule l'intermittence des connexions (échantillonnage) - O(n²) → O(n)"""
    try:
        if len(adj_sequence) < 3:
            return 0.0
        
        n = adj_sequence[0].shape[0]
        # ✅ OPTIMISATION : Échantillonner seulement sqrt(n²) = n paires au lieu de n²/2
        max_samples = min(100, n * n // 4)  # Limiter à 100 échantillons max
        
        intermittency_scores = []
        sample_count = 0
        
        # Échantillonner des paires aléatoirement
        for _ in range(max_samples):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            
            if i == j:
                continue
                
            edge_history = []
            for adj in adj_sequence:
                edge_history.append(int(adj[i, j].item() > 0))
            
            # Compter les changements d'état
            changes = sum(1 for k in range(1, len(edge_history)) 
                         if edge_history[k] != edge_history[k-1])
            
            # Normaliser par la longueur de la séquence
            intermittency = changes / (len(edge_history) - 1) if len(edge_history) > 1 else 0
            intermittency_scores.append(intermittency)
            sample_count += 1
            
            if sample_count >= max_samples:
                break
        
        return np.mean(intermittency_scores) if intermittency_scores else 0.0
        
    except Exception:
        return 0.0

def compute_degree_evolution(adj_sequence):
    """Calcule l'évolution du degré moyen"""
    try:
        if len(adj_sequence) < 2:
            return {'rate': 0.0, 'volatility': 0.0}
        
        degree_means = []
        for adj in adj_sequence:
            degrees = torch.sum(adj, dim=1)
            mean_degree = torch.mean(degrees).item()
            degree_means.append(mean_degree)
        
        # Taux de changement moyen
        changes = [degree_means[i+1] - degree_means[i] for i in range(len(degree_means)-1)]
        avg_rate = np.mean(changes) if changes else 0.0
        
        # Volatilité (écart-type des changements)
        volatility = np.std(changes) if len(changes) > 1 else 0.0
        
        return {'rate': avg_rate, 'volatility': volatility}
        
    except Exception:
        return {'rate': 0.0, 'volatility': 0.0}

def compute_density_evolution(adj_sequence):
    """Calcule l'évolution de la densité"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        densities = []
        for adj in adj_sequence:
            n = adj.shape[0]
            if n > 1:
                edge_count = torch.sum(adj).item() / 2
                max_edges = n * (n - 1) / 2
                density = edge_count / max_edges
            else:
                density = 0.0
            densities.append(density)
        
        # Changement moyen de densité
        changes = [densities[i+1] - densities[i] for i in range(len(densities)-1)]
        return np.mean(changes) if changes else 0.0
        
    except Exception:
        return 0.0

def compute_neighborhood_stability_fast(adj_sequence):
    """🚀 Calcule la stabilité du voisinage (échantillonnage) - O(n²) → O(n)"""
    try:
        if len(adj_sequence) < 2:
            return 1.0
        
        total_jaccard = 0
        total_comparisons = 0
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            n = prev_adj.shape[0]
            
            # ✅ OPTIMISATION : Échantillonner seulement sqrt(n) nœuds
            sample_size = min(50, max(5, int(np.sqrt(n))))
            sampled_nodes = random.sample(range(n), min(sample_size, n))
            
            node_jaccards = []
            for node in sampled_nodes:
                # Voisinages aux deux temps
                prev_neighbors = set(torch.where(prev_adj[node] > 0)[0].tolist())
                curr_neighbors = set(torch.where(curr_adj[node] > 0)[0].tolist())
                
                # Coefficient de Jaccard
                intersection = len(prev_neighbors & curr_neighbors)
                union = len(prev_neighbors | curr_neighbors)
                
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 1.0  # Si pas de voisins, stabilité parfaite
                
                node_jaccards.append(jaccard)
            
            avg_jaccard = np.mean(node_jaccards) if node_jaccards else 1.0
            total_jaccard += avg_jaccard
            total_comparisons += 1
        
        return total_jaccard / total_comparisons if total_comparisons > 0 else 1.0
        
    except Exception:
        return 0.0

def compute_clustering_evolution_fast(adj_sequence):
    """🚀 Calcule l'évolution du coefficient de clustering (heuristique rapide) - O(n³) → O(n)"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        clustering_coeffs = []
        for adj in adj_sequence:
            try:
                clustering = compute_clustering_coefficient_fast(adj)
                clustering_coeffs.append(clustering)
            except Exception:
                clustering_coeffs.append(0.0)
        
        # Changement moyen du clustering
        changes = [clustering_coeffs[i+1] - clustering_coeffs[i] for i in range(len(clustering_coeffs)-1)]
        return np.mean(changes) if changes else 0.0
        
    except Exception:
        return 0.0

def compute_clustering_coefficient_fast(adj_matrix):
    """🚀 Heuristique rapide pour le clustering - O(n³) → O(n)"""
    try:
        n = adj_matrix.shape[0]
        if n < 3 or torch.sum(adj_matrix) == 0:
            return 0.0
        
        # ✅ HEURISTIQUE : Échantillonner des nœuds au lieu de calculer exactement
        sample_size = min(50, max(5, int(np.sqrt(n))))
        sampled_nodes = random.sample(range(n), min(sample_size, n))
        
        total_clustering = 0
        valid_nodes = 0
        
        for node in sampled_nodes:
            # Voisins du nœud
            neighbors = torch.where(adj_matrix[node] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Compter les connexions entre voisins
            connections = 0
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if adj_matrix[neighbors[i], neighbors[j]] > 0:
                        connections += 1
            
            # Coefficient de clustering local
            possible_connections = k * (k - 1) / 2
            local_clustering = connections / possible_connections if possible_connections > 0 else 0
            
            total_clustering += local_clustering
            valid_nodes += 1
        
        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0
        
    except Exception:
        return 0.0

def compute_triangle_evolution_fast(adj_sequence):
    """🚀 Calcule l'évolution du nombre de triangles (approximation rapide) - O(n³) → O(n)"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        triangle_counts = []
        for adj in adj_sequence:
            try:
                count = count_triangles_fast(adj)
                triangle_counts.append(count)
            except Exception:
                triangle_counts.append(0)
        
        # Changement relatif moyen
        changes = []
        for i in range(len(triangle_counts)-1):
            prev_count = triangle_counts[i]
            curr_count = triangle_counts[i+1]
            
            if prev_count > 0:
                change = (curr_count - prev_count) / prev_count
            elif curr_count > 0:
                change = 1.0  # Apparition de triangles
            else:
                change = 0.0  # Pas de changement
            
            changes.append(change)
        
        return np.mean(changes) if changes else 0.0
        
    except Exception:
        return 0.0

def count_triangles_fast(adj_matrix):
    """🚀 Approximation rapide du nombre de triangles - O(n³) → O(n)"""
    try:
        n = adj_matrix.shape[0]
        if n < 3 or torch.sum(adj_matrix) == 0:
            return 0
        
        # ✅ HEURISTIQUE : Échantillonner des triplets au lieu de tous les calculer
        sample_size = min(100, max(10, n))
        triangle_count = 0
        
        for _ in range(sample_size):
            # Choisir 3 nœuds aléatoirement
            nodes = random.sample(range(n), min(3, n))
            if len(nodes) < 3:
                continue
                
            i, j, k = nodes[0], nodes[1], nodes[2]
            
            # Vérifier si c'est un triangle
            if (adj_matrix[i, j] > 0 and 
                adj_matrix[j, k] > 0 and 
                adj_matrix[k, i] > 0):
                triangle_count += 1
        
        # Extrapoler le nombre total de triangles
        total_possible_triplets = n * (n-1) * (n-2) / 6 if n >= 3 else 1
        estimated_triangles = triangle_count * (total_possible_triplets / sample_size)
        
        return int(max(0, estimated_triangles))
        
    except Exception:
        return 0

def compute_structural_autocorrelation(adj_sequence):
    """Calcule l'auto-corrélation structurelle"""
    try:
        if len(adj_sequence) < 3:
            return 0.0
        
        # Calculer les métriques structurelles pour chaque graphe
        metrics = []
        for adj in adj_sequence:
            try:
                density = torch.sum(adj).item() / (adj.shape[0] * (adj.shape[0] - 1))
                avg_degree = torch.mean(torch.sum(adj, dim=1)).item()
                clustering = compute_clustering_coefficient_fast(adj)  # ✅ Version rapide
                
                metrics.append([density, avg_degree, clustering])
            except Exception:
                metrics.append([0.0, 0.0, 0.0])
        
        # Calculer l'auto-corrélation avec décalage 1
        autocorrs = []
        for feature_idx in range(3):  # 3 métriques
            feature_values = [m[feature_idx] for m in metrics]
            
            if len(set(feature_values)) > 1:  # Variation dans les valeurs
                # Auto-corrélation simple
                correlations = []
                for i in range(len(feature_values)-1):
                    val1, val2 = feature_values[i], feature_values[i+1]
                    correlations.append(val1 * val2)
                
                autocorr = np.mean(correlations) if correlations else 0.0
            else:
                autocorr = 1.0 if feature_values[0] > 0 else 0.0
            
            autocorrs.append(autocorr)
        
        return np.mean(autocorrs)
        
    except Exception:
        return 0.0

def compute_structural_periodicity(adj_sequence):
    """Détecte la périodicité dans l'évolution structurelle"""
    try:
        if len(adj_sequence) < 4:
            return 0.0
        
        # Calculer une métrique simple : nombre d'arêtes
        edge_counts = []
        for adj in adj_sequence:
            count = torch.sum(adj).item() / 2
            edge_counts.append(count)
        
        # Détecter la périodicité en cherchant des patterns répétitifs
        max_period = min(len(edge_counts) // 2, 3)
        periodicity_scores = []
        
        for period in range(2, max_period + 1):
            correlations = []
            for i in range(len(edge_counts) - period):
                val1 = edge_counts[i]
                val2 = edge_counts[i + period]
                
                if val1 > 0 or val2 > 0:
                    correlation = 2 * min(val1, val2) / (val1 + val2 + 1e-8)
                else:
                    correlation = 1.0
                
                correlations.append(correlation)
            
            periodicity_scores.append(np.mean(correlations) if correlations else 0.0)
        
        return max(periodicity_scores) if periodicity_scores else 0.0
        
    except Exception:
        return 0.0

def compute_centrality_evolution(adj_sequence):
    """Calcule l'évolution de la centralité de degré"""
    try:
        if len(adj_sequence) < 2:
            return {'mean': 0.0, 'variance': 0.0}
        
        centrality_changes = []
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Centralités de degré
            prev_centrality = torch.sum(prev_adj, dim=1)
            curr_centrality = torch.sum(curr_adj, dim=1)
            
            # Changements de centralité
            changes = torch.abs(curr_centrality - prev_centrality)
            centrality_changes.extend(changes.tolist())
        
        mean_change = np.mean(centrality_changes) if centrality_changes else 0.0
        var_change = np.var(centrality_changes) if len(centrality_changes) > 1 else 0.0
        
        return {'mean': mean_change, 'variance': var_change}
        
    except Exception:
        return {'mean': 0.0, 'variance': 0.0}

def compute_centrality_persistence(adj_sequence):
    """Calcule la persistance des nœuds centraux"""
    try:
        if len(adj_sequence) < 2:
            return 1.0
        
        n = adj_sequence[0].shape[0]
        persistence_scores = []
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Identifier les nœuds les plus centraux (top 20% ou au moins 1)
            prev_degrees = torch.sum(prev_adj, dim=1)
            curr_degrees = torch.sum(curr_adj, dim=1)
            
            k = max(1, n // 5)  # Top 20%
            
            prev_top = set(torch.topk(prev_degrees, k)[1].tolist())
            curr_top = set(torch.topk(curr_degrees, k)[1].tolist())
            
            # Calcul de la persistance (intersection / union)
            intersection = len(prev_top & curr_top)
            union = len(prev_top | curr_top)
            
            persistence = intersection / union if union > 0 else 1.0
            persistence_scores.append(persistence)
        
        return np.mean(persistence_scores) if persistence_scores else 1.0
        
    except Exception:
        return 0.0

def compute_structural_entropy(adj_sequence):
    """Calcule l'entropie de l'évolution structurelle"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        # Calculer les distributions de degrés pour chaque graphe
        entropies = []
        
        for adj in adj_sequence:
            degrees = torch.sum(adj, dim=1)
            max_degree = torch.max(degrees).item()
            
            if max_degree > 0:
                # Distribution des degrés
                degree_counts = torch.bincount(degrees.int(), minlength=int(max_degree)+1)
                degree_probs = degree_counts.float() / torch.sum(degree_counts)
                
                # Entropie de Shannon
                entropy = 0.0
                for prob in degree_probs:
                    if prob > 0:
                        entropy -= prob * torch.log2(prob)
                
                entropies.append(entropy.item())
            else:
                entropies.append(0.0)
        
        # Variabilité de l'entropie comme mesure de complexité évolutive
        return np.std(entropies) if len(entropies) > 1 else 0.0
        
    except Exception:
        return 0.0

def compute_structural_surprise(adj_sequence):
    """Calcule la surprise structurelle (changements inattendus)"""
    try:
        if len(adj_sequence) < 3:
            return 0.0
        
        # Calculer les changements dans la densité
        densities = []
        for adj in adj_sequence:
            n = adj.shape[0]
            if n > 1:
                density = torch.sum(adj).item() / (n * (n - 1))
            else:
                density = 0.0
            densities.append(density)
        
        # Calculer les "surprises" comme écarts à la tendance
        surprises = []
        for i in range(2, len(densities)):
            # Tendance prédite basée sur les deux points précédents
            if i >= 2:
                predicted = 2 * densities[i-1] - densities[i-2]
                actual = densities[i]
                surprise = abs(actual - predicted)
                surprises.append(surprise)
        
        return np.mean(surprises) if surprises else 0.0
        
    except Exception:
        return 0.0

def compute_edge_predictability_fast(adj_sequence):
    """🚀 Calcule la prédictibilité de formation d'arêtes (échantillonnage) - O(n²) → O(n)"""
    try:
        if len(adj_sequence) < 3:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(2, len(adj_sequence)):
            prev_adj = adj_sequence[i-2]
            curr_adj = adj_sequence[i-1]
            next_adj = adj_sequence[i]
            
            n = prev_adj.shape[0]
            
            # ✅ OPTIMISATION : Échantillonner des paires au lieu de toutes les tester
            max_samples = min(100, n * n // 4)
            
            for _ in range(max_samples):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                
                if u == v:
                    continue
                    
                # Prédiction basée sur l'historique récent
                was_present = (prev_adj[u, v] > 0) or (curr_adj[u, v] > 0)
                is_present = (next_adj[u, v] > 0)
                
                if was_present == is_present:
                    correct_predictions += 1
                total_predictions += 1
                
                if total_predictions >= max_samples:
                    break
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
    except Exception:
        return 0.0

def compute_global_volatility(adj_sequence):
    """Calcule la volatilité globale du graphe"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        volatilities = []
        
        for i in range(1, len(adj_sequence)):
            prev_adj = adj_sequence[i-1]
            curr_adj = adj_sequence[i]
            
            # Différence entre les matrices d'adjacence
            diff = torch.abs(curr_adj - prev_adj)
            volatility = torch.sum(diff).item() / (curr_adj.shape[0] ** 2)
            volatilities.append(volatility)
        
        return np.mean(volatilities) if volatilities else 0.0
        
    except Exception:
        return 0.0

def compute_temporal_trend(adj_sequence):
    """Calcule la tendance temporelle (croissance/décroissance)"""
    try:
        if len(adj_sequence) < 2:
            return 0.0
        
        # Calculer l'évolution du nombre d'arêtes
        edge_counts = []
        for adj in adj_sequence:
            count = torch.sum(adj).item() / 2
            edge_counts.append(count)
        
        # Régression linéaire simple pour détecter la tendance
        if len(edge_counts) > 1:
            x = np.arange(len(edge_counts))
            y = np.array(edge_counts)
            
            # Pente de la régression linéaire
            if len(set(y)) > 1:  # Éviter division par zéro
                slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            else:
                slope = 0.0
            
            return slope
        else:
            return 0.0
        
    except Exception:
        return 0.0

class TemporalGraphFeatures(nn.Module):
    """
    Module PyTorch pour calculer les features temporelles de graphes dynamiques
    🚀 Optimisé : Complexité réduite de O(n³) à O(n) pour les opérations critiques
    """
    
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, graph_sequence, current_t, num_timesteps, device):
        """Interface principale pour l'entraînement"""
        return compute_temporal_graph_features(
            graph_sequence, current_t, num_timesteps, device, self.window_size
        )
    
    def get_feature_dim(self):
        """Retourne la dimension du vecteur de features temporelles"""
        return 22  # Nombre total de features temporelles calculées
    
    def set_window_size(self, window_size):
        """Configure la taille de la fenêtre temporelle"""
        self.window_size = window_size