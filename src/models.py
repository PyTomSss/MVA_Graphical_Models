import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_adjacency_and_features

from src.layer import GraphConvolutionLayer, GraphAttentionLayer, GraphDenseNetLayer
from src.pooling import pooling_op


class GCN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GCN, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.pool_type = pool_type

        # Tu peux utiliser ModuleList ici pour que PyTorch détecte bien les layers
        self.graph_convolution_layers = nn.ModuleList()
        for i in range(n_layer):
            in_dim = n_feat if i == 0 else agg_hidden
            self.graph_convolution_layers.append(GraphConvolutionLayer(in_dim, agg_hidden, device))
        
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x, adj):
        for i in range(self.n_layer):
            x = F.relu(self.graph_convolution_layers[i](x, adj))
            if i != self.n_layer - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = pooling_op(x, self.pool_type)  # Devrait retourner [B, agg_hidden] ou [agg_hidden]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    """
    def forward(self, data):
        x, adj = data[:2]

        for i in range(self.n_layer):
           # Graph convolution layer
           x = F.relu(self.graph_convolution_layers[i](x, adj))
                      
           # Dropout
           if i != self.n_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pool_type
        x = pooling_op(x, self.pool_type)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
    """
    

class GAT(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device, alpha=0.2):
        super(GAT, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.pool_type = pool_type
        
        # Initialisation des couches d'attention
        self.graph_attention_layers = nn.ModuleList()
        for i in range(self.n_layer):
            in_dim = n_feat if i == 0 else agg_hidden
            self.graph_attention_layers.append(GraphAttentionLayer(in_dim, agg_hidden, dropout, device, alpha))
        
        # Couches fully-connected
        self.fc1 = nn.Linear(agg_hidden * n_layer, fc_hidden)  # Multiplier par n_layer car on concatène les résultats
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x, adj):
        # Propagation à travers les couches d'attention
        x_layers = []
        for i in range(self.n_layer):
            # Application de chaque couche d'attention
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_layer = F.relu(self.graph_attention_layers[i](x, adj))
            x_layers.append(x_layer)

        # Concaténation des sorties de chaque couche d'attention
        x = torch.cat(x_layers, dim=2)  # Concatène sur la dimension des caractéristiques

        # Pooling (en fonction du type de pooling choisi)
        x = pooling_op(x, self.pool_type)  # Devrait retourner [B, agg_hidden] ou [agg_hidden]

        # Passage par les couches fully-connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    

class GraphDenseNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GraphDenseNet, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.pool_type = pool_type
        
        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(n_feat, agg_hidden, device)
        
        # Graph densenet layer
        self.graph_densenet_layers = []
        for i in range(self.n_layer):
            self.graph_densenet_layers.append(GraphDenseNetLayer(agg_hidden, agg_hidden, device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear((agg_hidden * 4), fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def forward(self, data):
        x, adj = data[:2]
        
        # Graph convolution layer
        x = F.relu(self.graph_convolution_layer(x, adj))

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.n_layer):
           pooling = False
           if i != 0: pooling = True
           
           # Graph densenet layer
           x = F.relu(self.graph_densenet_layers[i](x, adj, pooling))
                      
           # Dropout
           if i != self.n_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pool_type
        x = pooling_op(x, self.pool_type)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x