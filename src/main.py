import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import glob
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random

# Imports des modules du projet
from models.denoising_network import DenoisingNetwork
from models.transition_matrices import DiGressTransitionMatrices
from models.graph_features import compute_graph_features
from models.graph_temporel_features import compute_temporal_graph_features
from models.visualize_snapshot import SnapshotLoader, get_snapshot_files
# residual block and layernorme a optimize
# Déplacé ici :
class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, snapshot_files):
        self.dataset_name = dataset_name
        self.snapshot_files = snapshot_files
        
    def __len__(self):
        return len(self.snapshot_files)
        
    def __getitem__(self, idx):
        file_path = self.snapshot_files[idx]
        X = SnapshotLoader.get_X(file_path)
        E = SnapshotLoader.get_E(file_path)
        return X, E

#  NOUVEAU : Fonction de split aléatoire
def split_snapshots_random(snapshot_files, train_ratio=0.8):
    """Split aléatoire : mélange les snapshots"""
    snapshot_files_copy = snapshot_files.copy()  # Éviter de modifier l'original
    random.shuffle(snapshot_files_copy)
    
    total_snapshots = len(snapshot_files_copy)
    train_size = int(total_snapshots * train_ratio)
    
    train_files = snapshot_files_copy[:train_size]
    eval_files = snapshot_files_copy[train_size:]
    
    print(f"📊 Split aléatoire:")
    print(f"  Training: {len(train_files)} snapshots")
    print(f"  Évaluation: {len(eval_files)} snapshots")
    
    return train_files, eval_files

def compute_temporal_graph_features_simple(graph, current_t, num_timesteps, device):
    """Version simple : Répète le graphe pour simuler une séquence"""
    graph_sequence = [graph] * 3
    return compute_temporal_graph_features(graph_sequence, current_t, num_timesteps, device, window_size=3)

#  NOUVEAU : Fonction d'évaluation intégrée
def compute_threshold_vector(model_path, eval_snapshots, mode='dynamic'):
    """Calcule le vecteur de seuil sur les données d'évaluation"""
    print(f"Calcul du vecteur de seuil (mode: {mode})")
    
    # Configuration
    BATCH_SIZE = 1
    NUM_TIMESTEPS = 1000
    DEVICE = torch.device('cpu')
    
    # Charger le modèle
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Matrices de transition
    transition_matrices = DiGressTransitionMatrices(
        num_timesteps=NUM_TIMESTEPS,
        dataset_name='uci'
    ).to(DEVICE)
    
    categories_info = transition_matrices.get_categories_info()
    num_node_types = categories_info['num_X_categories']
    num_edge_types = categories_info['num_E_categories']
    
    # Modèle
    model = DenoisingNetwork(
        num_node_classes=num_node_types,
        num_edge_classes=num_edge_types,
        hidden_dim=64,
        num_layers=3,
        mode=mode
    ).to(DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Dataset d'évaluation
    eval_dataset = SnapshotDataset('uci', eval_snapshots)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_E) in enumerate(eval_dataloader):
            print(f" Évaluation batch {batch_idx+1}/{len(eval_dataloader)}")
            
            batch_X = batch_X.to(DEVICE)
            batch_E = batch_E.to(DEVICE)
            
            t = torch.randint(1, NUM_TIMESTEPS + 1, (batch_X.size(0),), device=DEVICE)
            
            # Bruitage (même processus que training)
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
            
            # Features
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
            
            #  UTILISER DIRECTEMENT LA LOSS DU MODÈLE
            loss = model(noisy_X, noisy_E, batch_features)
            reconstruction_errors.append(loss.item())
            
            print(f"  Loss de reconstruction: {loss.item():.6f}")
    
    # Calcul des statistiques de seuil
    if reconstruction_errors:
        errors = np.array(reconstruction_errors)
        threshold_vector = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'percentile_95_threshold': np.percentile(errors, 95),
            'percentile_99_threshold': np.percentile(errors, 99),
            'z_score_threshold_2': np.mean(errors) + 2 * np.std(errors),
            'sample_count': len(errors)
        }
        
        # Sauvegarder le vecteur de seuil
        os.makedirs("saved_models", exist_ok=True)
        threshold_path = f"saved_models/threshold_vector_{mode}_uci.pt"
        torch.save(threshold_vector, threshold_path)
        
        print(f" Vecteur de seuil sauvegardé: {threshold_path}")
        print(f" Seuil 95%: {threshold_vector['percentile_95_threshold']:.6f}")
        
        return threshold_vector
    else:
        print(" Aucune erreur calculée")
        return None

def train(mode, dataset='uci', num_epochs=100, model_path=None, snapshot_files=None):
    """
    Entraînement unifié -  MODIFIÉ pour accepter snapshot_files
    
    Args:
        mode: "static" ou "dynamic"
        dataset: nom du dataset
        num_epochs: nombre d'époques
        model_path: chemin du modèle pré-entraîné (None pour partir de zéro)
        snapshot_files: liste des fichiers snapshot à utiliser 
    """
    print(f"\n=== ENTRAÎNEMENT MODE {mode.upper()} ===")
    
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_TIMESTEPS = 1000
    DEVICE = torch.device('cpu')  # Forcer CPU pour éviter les problèmes de mémoire
    
    #  UTILISER LES SNAPSHOTS FOURNIS OU TOUS
    if snapshot_files is None:
        snapshot_files = get_snapshot_files(dataset)
    
    print(f"Nombre de snapshots pour training: {len(snapshot_files)}")
    
    # Obtenir le nombre total de catégories
    total_categories = SnapshotLoader.get_total_categories()
    print(f"Total categories: {total_categories}")
    
    # Afficher les distributions pour quelques snapshots seulement
    print("\n=== DISTRIBUTIONS (échantillon) ===")
    sample_files = snapshot_files[:min(3, len(snapshot_files))]  # Max 3 pour éviter spam
    for i, file_path in enumerate(sample_files):
        X = SnapshotLoader.get_X(file_path)
        E = SnapshotLoader.get_E(file_path)
        
        # Calculer les distributions pour ce snapshot
        X_counts = torch.zeros(6)
        E_counts = torch.zeros(2)
        
        # Distribution X
        for cat in range(total_categories):
            X_counts[cat] = (X[:, 1] == cat).sum().item()
        X_dist = X_counts / X_counts.sum()
        
        # Distribution E
        for cat in range(2):
            E_counts[cat] = (E == cat).sum().item()
        E_dist = E_counts / E_counts.sum()
        
        print(f"\nSnapshot {i}:")
        print(f"X distribution: {X_dist.tolist()}")
        print(f"E distribution: {E_dist.tolist()}")
    
    # Créer le dataset qui utilise SnapshotLoader
    dataset_obj = SnapshotDataset(dataset, snapshot_files)
    dataloader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=True)
    
    # Matrices de transition
    transition_matrices = DiGressTransitionMatrices(
        num_timesteps=NUM_TIMESTEPS,
        dataset_name='uci'
    ).to(DEVICE)
    
    categories_info = transition_matrices.get_categories_info()
    num_node_types = categories_info['num_X_categories']
    num_edge_types = categories_info['num_E_categories']
    
    # Modèle
    model = DenoisingNetwork(
        num_node_classes=num_node_types,
        num_edge_classes=num_edge_types,
        hidden_dim=64,
        num_layers=3,
        mode=mode
    ).to(DEVICE)
    
    # Chargement modèle pré-entraîné si fourni
    if model_path:
        print(f"Chargement du modèle: {model_path}")
        checkpoint = torch.load(model_path)
        model_state_dict = model.state_dict()
        
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        # Copier tous les poids sauf feature_embedding
        for key in pretrained_dict:
            if 'feature_embedding' not in key:
                if key in model_state_dict:
                    model_state_dict[key] = pretrained_dict[key]
        
        model.load_state_dict(model_state_dict)
        print("Poids chargés")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entraînement
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nÉpoque {epoch+1}/{num_epochs}")
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_X, batch_E) in enumerate(dataloader):
            print(f"\n Batch {batch_idx+1}/{len(dataloader)}")
            batch_X = batch_X.to(DEVICE)
            batch_E = batch_E.to(DEVICE)
            
            t = torch.randint(1, NUM_TIMESTEPS + 1, (batch_X.size(0),), device=DEVICE)
            
            # Bruitage
            print(f"\n PROCESSUS DE BRUITAGE - Batch {batch_idx+1}/{len(dataloader)}")
            print(f"   Timesteps échantillonnés: {t}")
            noisy_X_batch = []
            noisy_E_batch = []
            
            for b in range(batch_X.size(0)):
                class TempGraph:
                    def __init__(self, X, E):
                        self.X = X
                        self.E = E
                
                graph = TempGraph(batch_X[b], batch_E[b])
                print(f"\n   Traitement graphe {b+1}/{batch_X.size(0)}:")
                print(f"    - Timestep t = {t[b].item()}")
                print(f"    - Dimensions initiales:")
                print(f"      X: {graph.X.shape}")
                print(f"      E: {graph.E.shape}")
                
                noisy_graph = transition_matrices.apply_noise_to_graph(
                    graph, t[b].item(), DEVICE, graph.X.shape[0]
                )
                
                print(f"     Bruitage terminé:")
                print(f"      X bruité: {noisy_graph.X_onehot.shape}")
                print(f"      E bruité: {noisy_graph.E_onehot.shape}")
                
                
                noisy_X_batch.append(noisy_graph.X_onehot)
                noisy_E_batch.append(noisy_graph.E_onehot)
            
            noisy_X = torch.stack(noisy_X_batch)
            noisy_E = torch.stack(noisy_E_batch)
            
            print(f"\n  Statistiques finales du batch:")
            print(f"    - Batch noisy_X: {noisy_X.shape}")
            print(f"    - Batch noisy_E: {noisy_E.shape}")
            print(f"    - Moyenne noisy_X: {noisy_X.mean():.4f}")
            print(f"    - Moyenne noisy_E: {noisy_E.mean():.4f}")
            
            #  NOUVEAU: noisy_E est déjà au format one-hot [batch_size, num_edges, num_edge_classes]
            # Pas besoin de conversion supplémentaire
            noisy_E_onehot = noisy_E
            
            # Features selon le mode
            print(f"  Calcul des features ({mode})...")
            batch_features = []
            for i in range(batch_X.size(0)):
                class FeatureGraph:
                    def __init__(self, X, E):
                        self.X = X
                        self.E = E
                
                feature_graph = FeatureGraph(noisy_X[i], noisy_E_onehot[i])
                
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
            
            # Forward pass
            optimizer.zero_grad()
            print(f"   Forward pass...")
            loss = model(noisy_X, noisy_E_onehot, batch_features)
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Backward pass...")
            loss.backward()
            print(f"    Mise à jour des poids...")
            optimizer.step()
            print(f"  Batch terminé")
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"\rBatch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}", end="")
        
        avg_loss = total_loss / num_batches
        print(f"\nÉpoque {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Créer le dossier saved_models s'il n'existe pas
        os.makedirs("saved_models", exist_ok=True)
        
        # Sauvegarde du meilleur
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f"saved_models/digress_{mode}_best_{dataset}.pt"
            torch.save(model.state_dict(), best_model_path)
    
    # Sauvegarde finale
    final_path = f"saved_models/digress_{mode}_final_{dataset}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': best_loss,
        'mode': mode,
        'config': {
            'dataset': dataset,
            'num_node_classes': num_node_types,
            'num_edge_classes': num_edge_types
        }
    }, final_path)
    
    print(f"\nModèle sauvegardé: {final_path}")
    return final_path

def main():
    """Entraînement bi-phasé + Évaluation"""
    DATASET = 'uci'
    NUM_EPOCHS = 100
    
    #  Split des données
    all_snapshots = get_snapshot_files(DATASET)
    train_snapshots, eval_snapshots = split_snapshots_random(all_snapshots, train_ratio=0.8)
    
    # Phase 1: Static
    print(" PHASE 1: STATIC")
    static_model_path = train(mode="static", dataset=DATASET, num_epochs=NUM_EPOCHS, snapshot_files=train_snapshots)
    
    # Phase 2: Dynamic
    print(" PHASE 2: DYNAMIC")
    dynamic_model_path = train(mode="dynamic", dataset=DATASET, num_epochs=NUM_EPOCHS, model_path=static_model_path, snapshot_files=train_snapshots)
    
    #  Phase 3: Évaluation
    print("PHASE 3: ÉVALUATION")
    threshold_vector = compute_threshold_vector(dynamic_model_path, eval_snapshots, mode="dynamic")

    print(f"\n PIPELINE COMPLET TERMINÉ:")
    print(f" Modèle dynamic: {dynamic_model_path}")
    if threshold_vector:
        print(f" Seuil 95%: {threshold_vector['percentile_95_threshold']:.6f}")
        print(f" Échantillons évalués: {threshold_vector['sample_count']}")

if __name__ == "__main__":
    main()