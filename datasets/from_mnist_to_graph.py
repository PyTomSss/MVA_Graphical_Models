import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import distance_matrix

class MNISTGraphDataset(Dataset):
    def __init__(self, k=8):
        """
        Convertit le dataset MNIST en un graphe utilisable par un GNN.
        Chaque image est un noeud, avec des features dérivées des pixels.
        Les arêtes sont créées en connectant chaque noeud à ses k plus proches voisins.
        """
        # Charger MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        
        # Extraction des features (vectorisation des images)
        self.images = torch.stack([img[0].view(-1) for img in dataset])  # (60000, 28*28)
        self.labels = torch.tensor([img[1] for img in dataset])
        
        # Normalisation des features
        self.images = (self.images - self.images.mean(dim=0)) / (self.images.std(dim=0) + 1e-5)
        
        # Création de la matrice d'adjacence via KNN à la main
        dist_matrix = distance_matrix(self.images.numpy(), self.images.numpy())
        self.edge_index = self.build_knn_graph(dist_matrix, k)
    
    def build_knn_graph(self, dist_matrix, k):
        """Construit la matrice d'adjacence basée sur les k plus proches voisins."""
        num_nodes = dist_matrix.shape[0]
        edge_index = []
        for i in range(num_nodes):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]  # Exclut le noeud lui-même
            for neighbor in neighbors:
                edge_index.append((i, neighbor))
                edge_index.append((neighbor, i))  # Graphe non orienté
        return torch.tensor(edge_index, dtype=torch.long).T  # Shape (2, num_edges)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.edge_index