import json
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets.data import GraphData

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def get_max_num_nodes(dataset_str):
    import datasets
    dataset = getattr(datasets, dataset_str)()

    max_num_nodes = -1
    for d in dataset.dataset:
        max_num_nodes = max(max_num_nodes, d.num_nodes)
    return max_num_nodes

def visualise_graph(index, dataset):
    """
    Idea: Visualise 

    Args:
        index (int): index of the graph in the dataset we want to visualise
        dataset (IMDB or DD): dataset of graphs
    """

    data = Data(x=dataset[index].x, edge_index=dataset[index].edge_index)
    graph = to_networkx(data, to_undirected=True)

    # Visualisation
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42) 
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=100,
        cmap=plt.cm.viridis,
        alpha=0.9
    )

    plt.title(f"Graph Visualization - Index {index}", fontsize=15)
    plt.axis('off')  # Enlève les axes pour une vue plus propre
    plt.tight_layout()
    plt.show()


def normalize_adjacency(A):
    I = torch.eye(A.size(0)).to(A.device)
    A_hat = A + I  # Add self-loops
    D_hat = torch.diag(torch.sum(A_hat, dim=1) ** -0.5)
    return torch.mm(torch.mm(D_hat, A_hat), D_hat)  # D^(-1/2) * A * D^(-1/2)


def get_adjacency_and_features(graph_data):
    """
    Prend un élément du dataset (graph_data) et renvoie la matrice d'adjacence et les features associées.
    Assure que les indices dans edge_index sont valides.

    Args:
        graph_data : Un objet de type GraphData contenant les informations du graphe.

    Returns:
        adj_matrix : La matrice d'adjacence (tensor de taille [num_nodes, num_nodes]).
        features : Les features des nœuds (tensor de taille [num_nodes, feature_dim]).
    """
    # Extraire les features des nœuds
    features = graph_data.x  # Tensor de taille [num_nodes, feature_dim] 
    
    # Extraire les arêtes
    edge_index = graph_data.edge_index  # Tensor de taille [2, num_edges]

    num_nodes = len(features)  # Nombre de nœuds dans le graphe

    # let's renormalize edges with the right indexes to build more easily the adjency matrix
    min_idx = torch.min(torch.min(edge_index[0]), torch.min(edge_index[1]))

    # renomalize indexes
    edge_index = edge_index - min_idx 

    # Initialiser une matrice d'adjacence de taille [num_nodes, num_nodes]
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    for i in range(len(edge_index[0])):
        adj_matrix[edge_index[0][i], edge_index[1][i]] = 1

    A = normalize_adjacency(adj_matrix)

    return A, features

def create_batch_from_loader(batch_indices, x_dataset, y_dataset, device="cpu"):

    all_adjs, all_x, all_y, sizes = [], [], [], []
    node_offset = 0

    for idx in batch_indices:
        data = x_dataset[idx]
        y = y_dataset[idx]

        adj, x = get_adjacency_and_features(data)  # adj: [N, N], x: [N, F]
        N = x.shape[0]

        # Décalage dans la matrice d'adjacence
        all_adjs.append(F.pad(adj, (node_offset, 0, node_offset, 0)))  # pad gauche/haut
        all_x.append(x)
        all_y.append(torch.tensor([y], dtype=torch.long))
        sizes.append(N)
        node_offset += N

    x_batch = torch.cat(all_x, dim=0).to(device)                      # [N_total, F]
    adj_batch = torch.block_diag(*all_adjs).to(device)               # [N_total, N_total]
    y_batch = torch.cat(all_y, dim=0).to(device)                     # [B] ou [B, 1]

    return x_batch, adj_batch, y_batch, sizes
