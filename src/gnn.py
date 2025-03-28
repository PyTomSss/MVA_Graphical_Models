import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import GNNLayer
from pool import PoolingLayer


class GNN(nn.Module):
    def __init__(self, in_features, embedding_dim, out_features, num_layers=2, conv_type="gcn", task="node_prediction", pool_type="mean"):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.task = task
        self.conv_type = conv_type

        # List to hold the layers of the GNN
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GNNLayer(in_features, embedding_dim, conv_type=conv_type))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(embedding_dim, embedding_dim, conv_type=conv_type))

        # Output layer
        self.layers.append(GNNLayer(embedding_dim, out_features, conv_type=conv_type))

        self.pooling = PoolingLayer(pool_type=pool_type)

        # Définition de la couche finale selon la tâche
        if task == "node_prediction":
            self.final_layer = nn.Linear(out_features, 1)  
        elif task == "edge_prediction":
            self.final_layer = nn.Linear(out_features * 2, 1)  
        elif task == "graph_prediction":
            self.final_layer = nn.Linear(out_features, 1)  # Prédiction d'un seul score pour le graphe
        else:
            raise ValueError(f"Unknown task type {task}")

    def forward(self, x, adj_matrix=None):
        # Initial features
        h = x

        # Apply layers
        for i in range(self.num_layers):
            h_neighbors = self.aggregate_neighbors(h, adj_matrix)
            h = self.layers[i](h, h_neighbors)

        # Output layer
        if self.task == "node_prediction":
            output = self.final_layer(h)
        elif self.task == "edge_prediction":
            # Concatenate features of two nodes for edge prediction
            edge_features = self.aggregate_edge_features(h, adj_matrix)
            output = self.final_layer(edge_features)
        elif self.task == "graph_prediction":
            graph_embedding = self.pooling(h)  # Agrégation globale du graphe
            output = self.final_layer(graph_embedding)  # Un seul score pour le graphe
        return output
        

    def aggregate_neighbors(self, h, adj_matrix):
        # Aggregate neighboring nodes' features
        neighbors = torch.matmul(adj_matrix, h)
        return neighbors

    def aggregate_edge_features(self, h, adj_matrix):
        # Aggregate the features of node pairs for edge prediction
        edge_features = []
        for i in range(adj_matrix.size(0)):
            for j in range(i+1, adj_matrix.size(1)):
                if adj_matrix[i, j] == 1:  # If there's an edge
                    edge_features.append(torch.cat([h[i], h[j]], dim=0))
        return torch.stack(edge_features)

# Exemple d'utilisation du modèle
if __name__ == "__main__":
    # Exemple de caractéristiques de nœuds et matrice d'adjacence
    node_features = torch.randn(5, 10)  # 5 nœuds avec 10 caractéristiques chacun
    adj_matrix = torch.tensor([[0, 1, 0, 1, 0],
                               [1, 0, 1, 0, 0],
                               [0, 1, 0, 1, 1],
                               [1, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0]], dtype=torch.float)

    # Initialiser le GNN
    model = GNN(in_features=10, embedding_dim=16, out_features=8, num_layers=3, conv_type="gcn", task="node_prediction")
    output = model(node_features, adj_matrix)
    print(output)
