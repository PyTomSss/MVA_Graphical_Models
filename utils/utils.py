import json
import networkx as nx
#from torch_geometric.utils import to_networkx
#from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    plt.figure(figsize=(10, 7))
    nx.draw(graph, node_size=30, with_labels=False)
    plt.title(f"Graph Visualization - Index {index}")  # Add title here
    plt.show()


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

    return adj_matrix, features