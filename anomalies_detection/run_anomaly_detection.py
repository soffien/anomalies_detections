import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import glob
import torch.nn.functional as F
import numpy as np

# Ajouter le chemin vers src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.denoising_network import DenoisingNetwork
from models.transition_matrices import DiGressTransitionMatrices
from models.graph_features import compute_graph_features
from models.graph_temporel_features import compute_temporal_graph_features
from models.visualize_snapshot import SnapshotLoader, get_snapshot_files

def compute_temporal_graph_features_simple(graph, current_t, num_timesteps, device):
    """Version simple : Répète le graphe pour simuler une séquence"""
    graph_sequence = [graph] * 3
    return compute_temporal_graph_features(graph_sequence, current_t, num_timesteps, device, window_size=3)

def load_trained_model_and_threshold(dataset='uci', mode='dynamic'):
    """
    Charge le modèle entraîné ET le vecteur de seuil depuis saved_models/
    """
    print(f" Chargement du modèle et seuil pour {dataset} (mode: {mode})")
    
    # Chemins des fichiers sauvegardés
    model_path = f'saved_models/digress_{mode}_final_{dataset}.pt'
    threshold_path = f'saved_models/threshold_vector_{mode}_{dataset}.pt'
    
    # Vérifier que les fichiers existent
    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Modèle non trouvé: {model_path}\n Exécutez d'abord main.py pour entraîner le modèle")
    
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f" Vecteur de seuil non trouvé: {threshold_path}\n Exécutez d'abord main.py pour calculer le seuil")
    
    DEVICE = torch.device('cpu')
    
    # Charger le modèle
    print(f"Chargement modèle: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Matrices de transition
    transition_matrices = DiGressTransitionMatrices(
        num_timesteps=1000,
        dataset_name=dataset
    ).to(DEVICE)
    
    categories_info = transition_matrices.get_categories_info()
    num_node_types = categories_info['num_X_categories']
    num_edge_types = categories_info['num_E_categories']
    
    # Créer le modèle
    model = DenoisingNetwork(
        num_node_classes=num_node_types,
        num_edge_classes=num_edge_types,
        hidden_dim=64,
        num_layers=3,
        mode=mode
    ).to(DEVICE)
    
    # Charger les poids
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Modèle chargé avec config: {checkpoint.get('config', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print(f" Modèle chargé (format legacy)")
    
    model.eval()
    
    # Charger le vecteur de seuil
    print(f" Chargement seuil: {threshold_path}")
    threshold_vector = torch.load(threshold_path, map_location=DEVICE)
    print(f" Vecteur de seuil chargé (échantillons: {threshold_vector.get('sample_count', 'N/A')})")
    
    return model, transition_matrices, threshold_vector

def compute_single_graph_error(graph_X, graph_E, model, transition_matrices, mode='dynamic'):
    """
    Calcule l'erreur de reconstruction pour UN SEUL graphe
    """
    DEVICE = torch.device('cpu')
    NUM_TIMESTEPS = 1000
    
    # Préparer les données (même processus que main.py)
    batch_X = graph_X.unsqueeze(0).to(DEVICE) if graph_X.dim() == 2 else graph_X.to(DEVICE)
    batch_E = graph_E.unsqueeze(0).to(DEVICE) if graph_E.dim() == 2 else graph_E.to(DEVICE)
    
    print(f" Analyse graphe - X: {batch_X.shape}, E: {batch_E.shape}")
    
    t = torch.randint(1, NUM_TIMESTEPS + 1, (batch_X.size(0),), device=DEVICE)
    
    # Bruitage (MÊME PROCESSUS que main.py)
    noisy_X_batch = []
    noisy_E_batch = []
    
    for b in range(batch_X.size(0)):
        class TempGraph:
            def __init__(self, X, E):
                self.X = X
                self.E = E
        
        graph = TempGraph(batch_X[b], batch_E[b])
        noisy_graph = transition_matrices.apply_noise_to_graph(
            graph, t[b].item(), DEVICE, graph.X.shape[0]
        )
        
        noisy_X_batch.append(noisy_graph.X_onehot)
        noisy_E_batch.append(noisy_graph.E_onehot)
    
    noisy_X = torch.stack(noisy_X_batch)
    noisy_E = torch.stack(noisy_E_batch)
    
    # Features (MÊME PROCESSUS que main.py)
    batch_features = []
    for i in range(batch_X.size(0)):
        class FeatureGraph:
            def __init__(self, X, E):
                self.X = X
                self.E = E
        
        feature_graph = FeatureGraph(noisy_X[i], noisy_E[i])
        
        if mode == "static":
            features = compute_graph_features(
                feature_graph, t[i], NUM_TIMESTEPS, DEVICE
            )
        else:
            features = compute_temporal_graph_features_simple(
                feature_graph, t[i], NUM_TIMESTEPS, DEVICE
            )
        
        batch_features.append(features)
    
    batch_features = torch.stack(batch_features)
    
    # Calculer l'erreur de reconstruction (MÊME LOSS que main.py)
    with torch.no_grad():
        loss = model(noisy_X, noisy_E, batch_features)
        reconstruction_error = loss.item()
    
    print(f" Erreur de reconstruction calculée: {reconstruction_error:.6f}")
    return reconstruction_error

def detect_anomaly(graph_X, graph_E, dataset='uci', mode='dynamic', threshold_type='percentile_95_threshold'):
    """
    Fonction principale pour détecter les anomalies sur un nouveau graphe
    
    Args:
        graph_X: Tensor [num_nodes, features] - Données des nœuds
        graph_E: Tensor [num_edges, features] - Données des arêtes  
        dataset: nom du dataset pour charger le bon modèle
        mode: 'static' ou 'dynamic'
        threshold_type: type de seuil à utiliser
    
    Returns:
        dict: Résultats de détection
    """
    print(f"\n DÉTECTION D'ANOMALIE")
    print(f"Dataset: {dataset}, Mode: {mode}")
    
    # Charger modèle et seuil
    model, transition_matrices, threshold_vector = load_trained_model_and_threshold(dataset, mode)
    
    # Calculer l'erreur pour ce graphe
    reconstruction_error = compute_single_graph_error(
        graph_X, graph_E, model, transition_matrices, mode
    )
    
    # Comparer avec le seuil
    threshold = threshold_vector[threshold_type]
    is_anomaly = reconstruction_error > threshold
    
    # Calculer le score d'anomalie
    anomaly_score = reconstruction_error / threshold if threshold > 0 else float('inf')
    confidence = min(1.0, max(0.0, (reconstruction_error - threshold) / threshold)) if threshold > 0 else 1.0
    
    result = {
        'is_anomaly': bool(is_anomaly),
        'reconstruction_error': reconstruction_error,
        'threshold_used': threshold,
        'threshold_type': threshold_type,
        'anomaly_score': anomaly_score,
        'confidence': confidence
    }
    
    # Afficher le résultat
    print(f"\n RÉSULTAT DE DÉTECTION:")
    print(f"   Erreur: {result['reconstruction_error']:.6f}")
    print(f"   Seuil ({threshold_type}): {result['threshold_used']:.6f}")
    print(f"   Anomalie: {'OUI' if result['is_anomaly'] else 'NON'}")
    print(f"   Score: {result['anomaly_score']:.3f}")
    print(f"   Confiance: {result['confidence']:.3f}")
    
    return result

def detect_multiple_anomalies(graph_list, dataset='uci', mode='dynamic'):
    """
    Détecte les anomalies sur plusieurs graphes
    
    Args:
        graph_list: Liste de tuples [(X1, E1), (X2, E2), ...]
    
    Returns:
        Liste des résultats
    """
    print(f"\n🔍 DÉTECTION D'ANOMALIES MULTIPLES ({len(graph_list)} graphes)")
    
    # Charger une seule fois
    model, transition_matrices, threshold_vector = load_trained_model_and_threshold(dataset, mode)
    
    results = []
    
    for i, (graph_X, graph_E) in enumerate(graph_list):
        print(f"\n📊 Graphe {i+1}/{len(graph_list)}")
        
        # Calculer erreur
        error = compute_single_graph_error(graph_X, graph_E, model, transition_matrices, mode)
        
        # Détecter anomalie
        threshold = threshold_vector['percentile_95_threshold']
        is_anomaly = error > threshold
        
        result = {
            'graph_id': i,
            'is_anomaly': bool(is_anomaly),
            'reconstruction_error': error,
            'threshold_used': threshold,
            'anomaly_score': error / threshold if threshold > 0 else float('inf')
        }
        
        results.append(result)
        print(f"  {' ANOMALIE' if is_anomaly else ' NORMAL'} - Erreur: {error:.6f}")
    
    # Résumé
    anomalies_count = sum(1 for r in results if r['is_anomaly'])
    print(f"\n RÉSUMÉ: {anomalies_count}/{len(graph_list)} anomalies détectées")
    
    return results

def load_snapshot_as_graph(snapshot_path):
    """
    Charge un snapshot et le retourne sous forme de graphe (X, E)
    Utile pour tester sur de vrais snapshots
    """
    print(f" Chargement snapshot: {snapshot_path}")
    
    X = SnapshotLoader.get_X(snapshot_path)
    E = SnapshotLoader.get_E(snapshot_path)
    
    print(f"   X: {X.shape}, E: {E.shape}")
    
    return X, E

def test_on_real_snapshots(dataset='uci', mode='dynamic', num_test=5):
    """
    Test sur de vrais snapshots du dataset (pour validation)
    """
    print(f"\n TEST SUR VRAIS SNAPSHOTS ({num_test} échantillons)")
    
    # Obtenir quelques snapshots pour test
    all_snapshots = get_snapshot_files(dataset)
    test_snapshots = all_snapshots[-num_test:]  # Prendre les derniers
    
    results = []
    
    for i, snapshot_path in enumerate(test_snapshots):
        print(f"\n Test {i+1}/{len(test_snapshots)}: {os.path.basename(snapshot_path)}")
        
        # Charger le graphe
        X, E = load_snapshot_as_graph(snapshot_path)
        
        # Détecter anomalie
        result = detect_anomaly(X, E, dataset, mode)
        result['snapshot_path'] = snapshot_path
        results.append(result)
    
    return results

def main():
    """
    Exemples d'utilisation de la détection d'anomalies
    """
    print(" DÉTECTION D'ANOMALIES - EXEMPLES D'UTILISATION")
    
    # Vérifier que le modèle et seuil existent
    try:
        model, transition_matrices, threshold_vector = load_trained_model_and_threshold('uci', 'dynamic')
        print(f" Modèle et seuil chargés avec succès")
        print(f" Seuil 95%: {threshold_vector['percentile_95_threshold']:.6f}")
        
        # Exemple 1: Test sur vrais snapshots
        print("\n" + "="*50)
        print(" EXEMPLE 1: Test sur snapshots réels")
        real_results = test_on_real_snapshots('uci', 'dynamic', num_test=3)
        
        # Exemple 2: Graphe synthétique (pour test)
        print("\n" + "="*50) 
        print(" EXEMPLE 2: Graphe synthétique")
        
        # Créer un graphe de test simple
        test_X = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.float)  # 3 nœuds
        test_E = torch.tensor([[0, 1, 1], [1, 2, 1]], dtype=torch.float)     # 2 arêtes
        
        synthetic_result = detect_anomaly(test_X, test_E, 'uci', 'dynamic')
        
        print(f"\n Exemples terminés!")
        
    except FileNotFoundError as e:
        print(f" {e}")
        print(" Veuillez d'abord exécuter main.py pour entraîner le modèle et calculer le seuil")

if __name__ == "__main__":
    main()