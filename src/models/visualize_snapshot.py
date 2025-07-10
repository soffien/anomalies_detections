import torch
import os
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.simple_loader import SnapshotGenerator

def get_snapshot_files(dataset):
    """Retourne la liste des fichiers snapshot pour un dataset"""
    snapshot_dir = f"data/{dataset}/snapshots"
    snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.pt")))
    return snapshot_files


class SnapshotLoader:
    """Classe pour charger les éléments des fichiers snapshot.pt"""
    
    @staticmethod
    def get_X(snapshot_file: str):
        """Charge et retourne X depuis un fichier snapshot.pt"""
        snapshot_data = torch.load(snapshot_file)
        return snapshot_data['X']
    
    @staticmethod
    def get_E(snapshot_file: str):
        """Charge et retourne E depuis un fichier snapshot.pt"""
        snapshot_data = torch.load(snapshot_file)
        return snapshot_data['E']
    
    @staticmethod
    def get_total_categories():
        """Retourne le nombre total de catégories depuis SnapshotGenerator"""
        generator = SnapshotGenerator()
        return generator.get_total_categories_count()
    


# Exemple d'usage
if __name__ == "__main__":
    # Chemins vers les fichiers snapshots (relatif au répertoire racine)
    snapshot_path = "data/uci/snapshots/snapshot_0.pt"
    
    # Charger X
    X = SnapshotLoader.get_X(snapshot_path)
    print(f"X shape: {X.shape}")
    print(f"X: {X}")
    
    # Charger E  
    E = SnapshotLoader.get_E(snapshot_path)
    print(f"E shape: {E.shape}")
    print(f"E: {E}")
    
    # Charger nombre total de catégories (depuis SnapshotGenerator)
    total_categories = SnapshotLoader.get_total_categories()
    print(f"Total categories: {total_categories}")