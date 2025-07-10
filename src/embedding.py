"""
Embedding utilities for DiGress graph format conversion
Converts edge lists to adjacency matrices and prepares data for the GraphTransformer
"""
# ce fichier n'est pas utilisé dans le projet
import torch
import torch.nn.functional as F


def convert_edge_list_to_adjacency_matrix(E_list, num_nodes, num_edge_classes, device):
    """
    ✅ Convertit une liste d'arêtes en matrice d'adjacence one-hot
    E_list: [num_edges, 3] avec [source_id, destination_id, edge_type]
    Retourne: [num_nodes, num_nodes, num_edge_classes] one-hot
    """
    # Créer matrice d'adjacence vide
    adj_matrix = torch.zeros(num_nodes, num_nodes, num_edge_classes, device=device)
    
    # Remplir avec les arêtes existantes
    for i in range(E_list.shape[0]):
        src = E_list[i, 0].item()
        dst = E_list[i, 1].item()
        edge_type = E_list[i, 2].item()
        
        # S'assurer que les indices sont dans les bornes
        if src < num_nodes and dst < num_nodes and edge_type < num_edge_classes:
            adj_matrix[src, dst, edge_type] = 1.0
    
    return adj_matrix


def prepare_graph_for_digress(X, E, num_node_classes, num_edge_classes, device):
    """
    ✅ Prépare les données de graphe pour DiGress
    X: [num_nodes, 2] avec [node_id, node_category]
    E: [num_edges, 3] avec [source_id, destination_id, edge_type]
    
    Retourne:
    - X_onehot: [num_nodes, num_node_classes] one-hot
    - E_adjacency: [num_nodes, num_nodes, num_edge_classes] one-hot
    """
    num_nodes = X.shape[0]
    
    # Convertir X en one-hot
    node_categories = X[:, 1].long()  # [num_nodes]
    X_onehot = F.one_hot(node_categories, num_classes=num_node_classes).float().to(device)
    
    # Convertir E en matrice d'adjacence
    E_adjacency = convert_edge_list_to_adjacency_matrix(E, num_nodes, num_edge_classes, device)
    
    return X_onehot, E_adjacency


def prepare_batch_for_digress(batch_X, batch_E, num_node_classes, num_edge_classes, device):
    """
    ✅ Prépare un batch de graphes pour DiGress
    batch_X: [batch_size, num_nodes, 2]
    batch_E: [batch_size, num_edges, 3]
    
    Retourne:
    - batch_X_onehot: [batch_size, num_nodes, num_node_classes]
    - batch_E_adjacency: [batch_size, num_nodes, num_nodes, num_edge_classes]
    """
    batch_size = batch_X.shape[0]
    num_nodes = batch_X.shape[1]
    
    batch_X_onehot = []
    batch_E_adjacency = []
    
    for b in range(batch_size):
        X_onehot, E_adjacency = prepare_graph_for_digress(
            batch_X[b], batch_E[b], num_node_classes, num_edge_classes, device
        )
        batch_X_onehot.append(X_onehot)
        batch_E_adjacency.append(E_adjacency)
    
    # Stack en batch
    batch_X_onehot = torch.stack(batch_X_onehot)  # [batch_size, num_nodes, num_node_classes]
    batch_E_adjacency = torch.stack(batch_E_adjacency)  # [batch_size, num_nodes, num_nodes, num_edge_classes]
    
    return batch_X_onehot, batch_E_adjacency


def validate_digress_inputs(X_onehot, E_adjacency):
    """
    ✅ Valide que les inputs sont au bon format pour DiGress
    """
    # Vérifier les dimensions
    assert len(X_onehot.shape) == 3, f"X doit être 3D [batch, nodes, classes], reçu {X_onehot.shape}"
    assert len(E_adjacency.shape) == 4, f"E doit être 4D [batch, nodes, nodes, classes], reçu {E_adjacency.shape}"
    
    # Vérifier que c'est bien one-hot
    X_sums = torch.sum(X_onehot, dim=-1)
    E_sums = torch.sum(E_adjacency, dim=-1)
    
    assert torch.allclose(X_sums, torch.ones_like(X_sums), atol=1e-6), "X doit être en one-hot"
    assert torch.allclose(E_sums, torch.ones_like(E_sums), atol=1e-6), "E doit être en one-hot"
    
    # Vérifier la cohérence batch
    assert X_onehot.shape[0] == E_adjacency.shape[0], "Batch size incohérent"
    assert X_onehot.shape[1] == E_adjacency.shape[1], "Nombre de nœuds incohérent"
    assert E_adjacency.shape[1] == E_adjacency.shape[2], "E doit être carré"
    
    print(f"✅ Validation réussie:")
    print(f"   X: {X_onehot.shape}")
    print(f"   E: {E_adjacency.shape}")
    
    return True


def test_embedding_conversion():
    """
    ✅ Test de la conversion d'embedding
    """
    print("🧪 Test de conversion d'embedding...")
    
    # Données de test
    device = torch.device('cpu')
    num_nodes = 5
    num_node_classes = 6
    num_edge_classes = 2
    
    # X: [num_nodes, 2] avec [node_id, node_category]
    X = torch.tensor([
        [0, 2],  # nœud 0, catégorie 2
        [1, 3],  # nœud 1, catégorie 3
        [2, 1],  # nœud 2, catégorie 1
        [3, 4],  # nœud 3, catégorie 4
        [4, 0],  # nœud 4, catégorie 0
    ])
    
    # E: [num_edges, 3] avec [source_id, destination_id, edge_type]
    E = torch.tensor([
        [0, 1, 1],  # arête 0→1, type 1
        [1, 2, 1],  # arête 1→2, type 1
        [2, 3, 0],  # arête 2→3, type 0
        [3, 4, 1],  # arête 3→4, type 1
        [0, 4, 0],  # arête 0→4, type 0
    ])
    
    # Conversion
    X_onehot, E_adjacency = prepare_graph_for_digress(X, E, num_node_classes, num_edge_classes, device)
    
    print(f"X original: {X.shape}")
    print(f"X one-hot: {X_onehot.shape}")
    print(f"E original: {E.shape}")
    print(f"E adjacency: {E_adjacency.shape}")
    
    # Validation
    validate_digress_inputs(X_onehot.unsqueeze(0), E_adjacency.unsqueeze(0))
    
    print("✅ Test réussi !")
    return X_onehot, E_adjacency


if __name__ == "__main__":
    test_embedding_conversion() 