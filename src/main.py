import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import glob
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random
import gc

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

#  NOUVEAU : Fonction d'évaluation intégrée avec optimisations mémoire
def compute_threshold_vector(model_path, eval_snapshots, mode='dynamic'):
    """Calcule le vecteur de seuil sur les données d'évaluation avec optimisations mémoire"""
    print(f"💾 Calcul du vecteur de seuil (mode: {mode}) avec optimisations mémoire")
    
    # Configuration optimisée
    BATCH_SIZE = 2  # 🔥 RÉDUIT ENCORE PLUS pour l'évaluation
    NUM_TIMESTEPS = 1000
    DEVICE = torch.device('cpu')
    
    # Vider le cache mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
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
    
    # Dataset d'évaluation avec batch size réduit
    eval_dataset = SnapshotDataset('uci', eval_snapshots)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_E) in enumerate(eval_dataloader):
            print(f" 📊 Évaluation batch {batch_idx+1}/{len(eval_dataloader)}")
            
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_E = batch_E.to(DEVICE, non_blocking=True)
            
            t = torch.randint(1, NUM_TIMESTEPS + 1, (batch_X.size(0),), device=DEVICE)
            
            # Bruitage (même processus que training mais avec autocast)
            with autocast():
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
                
                # Libérer la mémoire intermédiaire
                del noisy_X_batch, noisy_E_batch
                gc.collect()
                
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
            
            # Libération agressive de la mémoire
            del batch_X, batch_E, noisy_X, noisy_E, batch_features, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
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
        
        print(f" ✅ Vecteur de seuil sauvegardé: {threshold_path}")
        print(f" 🎯 Seuil 95%: {threshold_vector['percentile_95_threshold']:.6f}")
        
        return threshold_vector
    else:
        print(" ❌ Aucune erreur calculée")
        return None

def train(mode, dataset='uci', num_epochs=100, model_path=None, snapshot_files=None):
    """
    Entraînement unifié avec optimisations mémoire pour CPU
    
    Args:
        mode: "static" ou "dynamic"
        dataset: nom du dataset
        num_epochs: nombre d'époques
        model_path: chemin du modèle pré-entraîné (None pour partir de zéro)
        snapshot_files: liste des fichiers snapshot à utiliser 
    """
    print(f"\n=== ENTRAÎNEMENT MODE {mode.upper()} AVEC OPTIMISATIONS MÉMOIRE CPU ===")
    
    # Configuration avec optimisations mémoire pour CPU
    BATCH_SIZE = 4  # 🔥 RÉDUIT DE 32 À 4 pour économiser la RAM
    LEARNING_RATE = 0.001
    NUM_TIMESTEPS = 1000
    DEVICE = torch.device('cpu')  # CPU uniquement
    
    # 🚀 NOUVEAU: Mixed precision seulement si CUDA disponible
    use_mixed_precision = torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None
    
    print(f"💾 Configuration optimisée pour la mémoire:")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size réduit: {BATCH_SIZE}")
    print(f"   Mixed precision: {'Activé (CUDA)' if use_mixed_precision else 'Désactivé (CPU)'}")
    print(f"   Gradient accumulation: Activé")
    
    #  UTILISER LES SNAPSHOTS FOURNIS OU TOUS
    if snapshot_files is None:
        snapshot_files = get_snapshot_files(dataset)
    
    print(f"Nombre de snapshots pour training: {len(snapshot_files)}")
    
    # Vider le cache mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Obtenir le nombre total de catégories
    total_categories = SnapshotLoader.get_total_categories()
    print(f"Total categories: {total_categories}")
    
    # Afficher les distributions pour quelques snapshots seulement
    print("\n=== DISTRIBUTIONS (échantillon) ===")
    sample_files = snapshot_files[:min(2, len(snapshot_files))]  # Réduit à 2 échantillons
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
        
        # Libérer la mémoire immédiatement
        del X, E, X_counts, E_counts
        gc.collect()
    
    # Créer le dataset qui utilise SnapshotLoader
    dataset_obj = SnapshotDataset(dataset, snapshot_files)
    dataloader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    
    # Matrices de transition
    transition_matrices = DiGressTransitionMatrices(
        num_timesteps=NUM_TIMESTEPS,
        dataset_name='uci'
    ).to(DEVICE)
    
    categories_info = transition_matrices.get_categories_info()
    num_node_types = categories_info['num_X_categories']
    num_edge_types = categories_info['num_E_categories']
    
    # Modèle (garder en FP32 pour CPU)
    model = DenoisingNetwork(
        num_node_classes=num_node_types,
        num_edge_classes=num_edge_types,
        hidden_dim=64,
        num_layers=3,
        mode=mode
    ).to(DEVICE)
    
    # 🚀 CORRECTION: NE PAS convertir en half precision sur CPU
    print(f"💾 Modèle configuré en FP32 pour CPU")
    
    # Chargement modèle pré-entraîné si fourni
    if model_path:
        print(f"Chargement du modèle: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
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
    
    # 🚀 NOUVEAU: Gradient accumulation pour simuler des batch plus gros
    accumulation_steps = 8  # Simule batch_size * 8 = 32
    
    # Entraînement
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nÉpoque {epoch+1}/{num_epochs}")
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Reset de l'optimiseur pour l'accumulation
        optimizer.zero_grad()
        
        for batch_idx, (batch_X, batch_E) in enumerate(dataloader):
            print(f"\n💾 Batch {batch_idx+1}/{len(dataloader)}")
            
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_E = batch_E.to(DEVICE, non_blocking=True)
            
            t = torch.randint(1, NUM_TIMESTEPS + 1, (batch_X.size(0),), device=DEVICE)
            
            # 🚀 CORRECTION: Forward pass avec ou sans autocast selon le device
            if use_mixed_precision and torch.cuda.is_available():
                context_manager = autocast('cuda')  # Utiliser la nouvelle API
            else:
                context_manager = torch.no_grad()  # Pas d'autocast sur CPU
                context_manager = torch.enable_grad()  # Mais on veut les gradients
            
            # Bruitage (processus simplifié pour éviter les problèmes de dtype)
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
            
            # Libérer la mémoire intermédiaire
            del noisy_X_batch, noisy_E_batch
            gc.collect()
            
            noisy_E_onehot = noisy_E
            
            # Features selon le mode
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
            
            # Forward pass du modèle (sans autocast sur CPU)
            loss = model(noisy_X, noisy_E_onehot, batch_features)
            
            # 🚀 NOUVEAU: Normaliser la loss pour l'accumulation
            loss = loss / accumulation_steps
            
            # 🚀 CORRECTION: Backward pass avec ou sans gradient scaling
            if use_mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 🚀 NOUVEAU: Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Mise à jour des poids
                if use_mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad()
                print(f"   ✅ Poids mis à jour (accumulation: {accumulation_steps} steps)")
            
            total_loss += loss.item() * accumulation_steps  # Dénormaliser pour l'affichage
            num_batches += 1
            
            print(f"   Loss: {loss.item() * accumulation_steps:.4f}")
            
            # Libération agressive de la mémoire
            del batch_X, batch_E, noisy_X, noisy_E, batch_features, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if batch_idx % 5 == 0:  # Réduire la fréquence d'affichage
                print(f"\rBatch {batch_idx+1}/{len(dataloader)} | Loss: {total_loss/num_batches:.4f}", end="")
        
        avg_loss = total_loss / num_batches
        print(f"\nÉpoque {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Créer le dossier saved_models s'il n'existe pas
        os.makedirs("saved_models", exist_ok=True)
        
        # Sauvegarde du meilleur
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f"saved_models/digress_{mode}_best_{dataset}.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Nouveau meilleur modèle sauvegardé: {best_loss:.4f}")
    
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
    
    print(f"\n✅ Modèle sauvegardé: {final_path}")
    print(f"🎯 Meilleure loss: {best_loss:.4f}")
    return final_path

def main():
    """Entraînement bi-phasé + Évaluation avec optimisations mémoire"""
    DATASET = 'uci'
    NUM_EPOCHS = 10  # 🔥 RÉDUIT POUR TESTS (passer à 100 quand ça marche)
    
    print(f"🚀 PIPELINE D'ENTRAÎNEMENT OPTIMISÉ POUR LA MÉMOIRE")
    print(f"   Dataset: {DATASET}")
    print(f"   Époques: {NUM_EPOCHS}")
    
    # Vider le cache au début
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    #  Split des données
    all_snapshots = get_snapshot_files(DATASET)
    
    # 🔥 NOUVEAU: Limiter le nombre de snapshots pour les tests
    max_snapshots = 20  # Limiter à 20 snapshots pour éviter OOM
    if len(all_snapshots) > max_snapshots:
        print(f"⚠️  Limitation des snapshots à {max_snapshots} pour éviter les problèmes de mémoire")
        all_snapshots = all_snapshots[:max_snapshots]
    
    train_snapshots, eval_snapshots = split_snapshots_random(all_snapshots, train_ratio=0.8)
    
    # Vérifier qu'on a des données
    if len(train_snapshots) == 0:
        print("❌ Aucun snapshot de training disponible!")
        return
    
    if len(eval_snapshots) == 0:
        print("⚠️  Aucun snapshot d'évaluation, utilisation d'un subset du training")
        eval_snapshots = train_snapshots[:2]  # Prendre 2 snapshots pour l'éval
    
    # Phase 1: Static
    print("\n🏗️  PHASE 1: STATIC")
    try:
        static_model_path = train(mode="static", dataset=DATASET, num_epochs=NUM_EPOCHS, snapshot_files=train_snapshots)
        print(f"✅ Phase statique terminée: {static_model_path}")
        
        # Vider la mémoire entre les phases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Erreur en phase statique: {e}")
        return
    
    # Phase 2: Dynamic
    print("\n🔄 PHASE 2: DYNAMIC")
    try:
        dynamic_model_path = train(mode="dynamic", dataset=DATASET, num_epochs=NUM_EPOCHS, model_path=static_model_path, snapshot_files=train_snapshots)
        print(f"✅ Phase dynamique terminée: {dynamic_model_path}")
        
        # Vider la mémoire avant l'évaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Erreur en phase dynamique: {e}")
        return
    
    #  Phase 3: Évaluation
    print("\n📊 PHASE 3: ÉVALUATION")
    try:
        threshold_vector = compute_threshold_vector(dynamic_model_path, eval_snapshots, mode="dynamic")
        
        print(f"\n🎉 PIPELINE COMPLET TERMINÉ:")
        print(f"   📁 Modèle dynamic: {dynamic_model_path}")
        if threshold_vector:
            print(f"   🎯 Seuil 95%: {threshold_vector['percentile_95_threshold']:.6f}")
            print(f"   📊 Échantillons évalués: {threshold_vector['sample_count']}")
        else:
            print(f"   ⚠️  Évaluation incomplète")
            
    except Exception as e:
        print(f"❌ Erreur en phase d'évaluation: {e}")
        print(f"   Le modèle est quand même sauvegardé: {dynamic_model_path}")

    # Nettoyage final
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\n💾 Nettoyage mémoire terminé")

if __name__ == "__main__":
    main()