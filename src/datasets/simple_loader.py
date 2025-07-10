import numpy as np
import os
import torch

# modifier la methode d'enregistrement en un fichier .pt
# compter dynamically les catégories de nœuds dans chaque snapshot
# et le nombre total de catégories possibles
# annuler le stockage des snapshots de graphes dynamiques (remplacer les methodes a haute complexité par des heurestiques ou des mlp apprenables)
class SnapshotGenerator:
    """Générateur de snapshots pour datasets de graphes dynamiques"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir     
        self.total_categories = 6
    
    def _load_dataset(self, dataset: str):
        """Charge un dataset et retourne (nœuds, arêtes, timestamps, unique_edges)"""
        if dataset == 'digg' or dataset == 'uci':
            file_path = os.path.join(self.data_dir, 'digg.txt')
            data = np.loadtxt(file_path, dtype=int, comments='%')
            edges = data[:, 0:2]
            timestamps = data[:, 3].astype(int) 
        elif dataset == 'bitcoinalpha' or dataset == 'bitcoinotc':
            # Pour bitcoinalpha et bitcoinotc, on charge les données CSV
            file_path = os.path.join(self.data_dir, 'soc-sign-bitcoinalpha.csv')
            with open(file_path) as f:
                lines = f.read().splitlines()
            data = np.array([[float(r) for r in row.split(',')] for row in lines])
            edges = data[:, 0:2].astype(int)
            timestamps = data[:, 3].astype(int)
        else:
            raise ValueError(f"Dataset {dataset} non reconnu")
        
        # Extraire tous les nœuds uniques
        all_nodes = set()
        all_edges_unique = set()
        for edge in edges:
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])
            all_edges_unique.add((edge[0], edge[1]))
        nodes = list(all_nodes)
        unique_edges = list(all_edges_unique)
        
        return nodes, edges, timestamps, unique_edges

    def _create_snapshot_tensors(self, snapshot_edges, global_nodes, unique_edges):
        """Crée X et E pour un snapshot"""
        # X : tensor des nœuds [N, 2]
        X = torch.zeros((len(global_nodes), 2), dtype=torch.long)
        for i, node in enumerate(global_nodes):
            X[i, 0] = node
            X[i, 1] = 0  # Inexistant par défaut
        
        # E : tensor de tous les edges du graphe [M, 3]
        E = torch.zeros((len(unique_edges), 3), dtype=torch.long)
        for i, edge in enumerate(unique_edges):
            E[i, 0] = edge[0]  # Source
            E[i, 1] = edge[1]  # Destination
            E[i, 2] = 0       # Toutes à zéro par défaut
        
        # Créer set des arêtes existantes dans ce snapshot
        existing_edges = set()
        for edge in snapshot_edges:
            existing_edges.add((edge[0], edge[1]))
        
        # Marquer les edges qui existent dans ce snapshot
        for i, edge in enumerate(unique_edges):
            if edge in existing_edges:
                E[i, 2] = 1
        
        # Calculer degrés pour les noeuds de ce snapshot
        degrees = {}
        for node in global_nodes:
            degrees[node] = 0
        
        for edge in snapshot_edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        # Quartiles des degrés existants
        existing_degrees = [d for d in degrees.values() if d > 0]
        if existing_degrees:
            existing_degrees.sort()
            n = len(existing_degrees)
            q1 = existing_degrees[n//4] if n > 3 else existing_degrees[0]
            q2 = existing_degrees[n//2] if n > 1 else existing_degrees[0]
            q3 = existing_degrees[3*n//4] if n > 3 else existing_degrees[-1]
            
            # Assigner catégories
            for i, node in enumerate(global_nodes):
                deg = degrees[node]
                if deg == 0 and node in [e[0] for e in snapshot_edges] + [e[1] for e in snapshot_edges]:
                    X[i, 1] = 1
                elif deg > 0:
                    if 1 <= deg <= q1:
                        X[i, 1] = 2
                    elif q1 < deg <= q2:
                        X[i, 1] = 3
                    elif q2 < deg <= q3:
                        X[i, 1] = 4
                    else:
                        X[i, 1] = 5
        
        return X, E

    def _process_dataset(self, dataset: str):
        """Génère et sauvegarde tous les snapshots d'un dataset"""
        # Charger dataset
        global_nodes, all_edges, timestamps, unique_edges = self._load_dataset(dataset)
        
        # Créer dossier et supprimer anciens fichiers
        output_dir = os.path.join(self.data_dir, dataset, 'snapshots')
        os.makedirs(output_dir, exist_ok=True)
        for existing_file in os.listdir(output_dir):
            if existing_file.endswith('.pt'):
                os.remove(os.path.join(output_dir, existing_file))
        
        # Calculer fenêtres temporelles
        unique_timestamps = len(set(timestamps))
        window_size = unique_timestamps // 10
        step_size = unique_timestamps // 50
        
        # Trier par timestamps
        sorted_indices = np.argsort(timestamps)
        sorted_edges = all_edges[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]
        
        # Générer snapshots par fenêtres temporelles
        unique_times = sorted(set(timestamps))
        snapshot_id = 0
        
        for start_time_idx in range(0, len(unique_times) - window_size + 1, step_size):
            end_time_idx = start_time_idx + window_size
            start_time = unique_times[start_time_idx]
            end_time = unique_times[end_time_idx - 1]
            
            # Sélectionner arêtes dans cette fenêtre temporelle
            mask = (sorted_timestamps >= start_time) & (sorted_timestamps <= end_time)
            snapshot_edges = sorted_edges[mask]
            
            if len(snapshot_edges) > 0:
                X, E = self._create_snapshot_tensors(snapshot_edges, global_nodes, unique_edges)
                
                snapshot_data = {
                    'snapshot_id': snapshot_id,
                    'X': X,
                    'E': E,
                    'num_categories_snapshot': len(torch.unique(X[:, 1])),
                    'total_categories': self.total_categories
                }
                
                torch.save(snapshot_data, os.path.join(output_dir, f'snapshot_{snapshot_id}.pt'))
                print(f"Snapshot {snapshot_id} sauvegardé pour {dataset}")
                snapshot_id += 1

    def get_snapshot_categories_count(self, snapshot_file: str) -> int:
        """Retourne le nombre de catégories dans un snapshot"""
        return torch.load(snapshot_file)['num_categories_snapshot']

    def get_total_categories_count(self) -> int:
        """Retourne le nombre total de catégories possibles"""
        return self.total_categories

    def process_all_datasets(self):
        """Traite tous les datasets disponibles"""
        for dataset in ['uci', 'digg', 'bitcoinalpha', 'bitcoinotc']:
            print(f"Traitement du dataset {dataset}...")
            self._process_dataset(dataset)
            print(f"Dataset {dataset} traité avec succès")

if __name__ == "__main__":
    generator = SnapshotGenerator()
    generator.process_all_datasets()