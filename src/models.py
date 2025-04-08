# ==============================================================================
# File: models.py
#
# Description:
# This file contains the implementation of several graph neural network models:
# GCN, GAT, GraphSAGE, GraphDenseNet, and GIN.
# These models are used for supervised learning on graph-structured data (e.g., 
# graph or node classification).
#
# Each model follows a similar structure: a series of layers for graph propagation 
# (convolutions, attention mechanisms, etc.), one or more pooling layers, and 
# fully-connected layers to produce final predictions.
# ==============================================================================
 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.layer import GraphConvolutionLayer, GraphAttentionLayer, GraphDenseNetLayer, GINLayer
from src.pooling import pooling_op

# ==============================================================================
# Class: GCN (Graph Convolutional Network)
#
# Description:
# This class implements a Graph Convolutional Network (GCN) model for graph-based 
# node classification tasks. The model applies multiple graph convolution layers 
# to aggregate information from neighboring nodes and then performs classification 
# using fully connected layers.
# ==============================================================================
class GCN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GCN, self).__init__()
        
        self.n_layer = n_layer  # Number of graph convolution layers
        self.dropout = dropout  # Dropout rate for regularization
        self.pool_type = pool_type  # Type of pooling operation to be used

        # Initialize graph convolution layers
        self.graph_convolution_layers = nn.ModuleList()
        for i in range(n_layer):
            # Input dimension for the first layer is n_feat, otherwise it's agg_hidden
            in_dim = n_feat if i == 0 else agg_hidden
            self.graph_convolution_layers.append(GraphConvolutionLayer(in_dim, agg_hidden, device))
        
        # Fully connected layers for final classification
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x, adj):
        # Forward pass through the graph convolution layers
        for i in range(self.n_layer):
            x = F.relu(self.graph_convolution_layers[i](x, adj))  # Apply ReLU activation after each convolution
            if i != self.n_layer - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout (except for the last layer)

        # Apply pooling operation (e.g., global average pooling) after convolutions
        x = pooling_op(x, self.pool_type)  # Output should be of shape [B, agg_hidden] or [agg_hidden]
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==============================================================================
# Class: GAT (Graph Attention Network)
#
# Description:
# This class implements a Graph Attention Network (GAT) model that uses attention 
# mechanisms to aggregate information from neighboring nodes, with learnable attention 
# coefficients. The model is designed for graph-based node classification.
# ==============================================================================
class GAT(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device, alpha=0.2):
        super(GAT, self).__init__()

        self.n_layer = n_layer  # Number of graph attention layers
        self.dropout = dropout  # Dropout rate for regularization
        self.pool_type = pool_type  # Type of pooling operation to be used
        
        # Initialize graph attention layers
        self.graph_attention_layers = nn.ModuleList()
        for i in range(self.n_layer):
            in_dim = n_feat if i == 0 else agg_hidden
            self.graph_attention_layers.append(GraphAttentionLayer(in_dim, agg_hidden, dropout, device, alpha))
        
        # Fully connected layers for final classification
        self.fc1 = nn.Linear(agg_hidden * n_layer, fc_hidden)  # Concatenate results from each layer
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x, adj):
        # Forward pass through graph attention layers
        x_layers = []
        for i in range(self.n_layer):
            # Apply attention layer with dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_layer = F.relu(self.graph_attention_layers[i](x, adj))
            x_layers.append(x_layer)
            x = x_layer

        # Concatenate outputs from each attention layer
        x = torch.cat(x_layers, dim=1)  # Concatenate along the feature dimension

        # Apply pooling operation (e.g., global average pooling)
        x = pooling_op(x, self.pool_type)  # Output should be of shape [B, agg_hidden] or [agg_hidden]

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# ==============================================================================
# Class: GraphSAGE (Graph Sample and Aggregation)
#
# Description:
# This class implements the GraphSAGE model, which aggregates features from a 
# fixed-size sampled neighborhood. The model is suitable for large-scale graphs 
# where it is computationally expensive to use the full neighborhood for aggregation.
# ==============================================================================
class GraphSAGE(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GraphSAGE, self).__init__()

        self.n_layer = n_layer  # Number of GraphSAGE layers
        self.dropout = dropout  # Dropout rate for regularization
        self.pool_type = pool_type  # Type of pooling operation to be used

        # Initialize GraphSAGE convolution layers
        self.graphsage_layers = nn.ModuleList()
        for i in range(n_layer):
            in_dim = n_feat if i == 0 else agg_hidden
            self.graphsage_layers.append(SAGEConv(in_dim, agg_hidden))

        # Fully connected layers for final classification
        self.fc1 = nn.Linear(agg_hidden * n_layer, fc_hidden)  # Concatenate results from each layer
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x_graph_data, batch=None):
        # Forward pass through GraphSAGE layers
        edge_index = x_graph_data.edge_index
        min_idx = torch.min(torch.min(edge_index[0]), torch.min(edge_index[1]))
        edge_index = x_graph_data.edge_index - min_idx
        x = x_graph_data.x

        x_layers = []
        for i in range(self.n_layer):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.graphsage_layers[i](x, edge_index))
            x_layers.append(x)

        # Concatenate outputs from each layer
        x = torch.cat(x_layers, dim=1)  # Concatenate along the feature dimension

        # Apply pooling operation (e.g., global average pooling)
        x = pooling_op(x, self.pool_type, batch)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==============================================================================
# Class: GIN (Graph Isomorphism Network)
#
# Description:
# This model implements the Graph Isomorphism Network (GIN), which aims to better distinguish 
# between non-isomorphic graphs by aggregating neighbor information in a more expressive manner.
# GIN is effective for graph classification tasks and is designed to be more powerful than GCNs.
# ==============================================================================
class GIN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GIN, self).__init__()

        self.n_layer = n_layer  # Number of GIN layers
        self.dropout = dropout  # Dropout rate for regularization
        self.pool_type = pool_type  # Type of pooling operation to be used
        self.device = device

        # Initialize GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(n_layer):
            in_dim = n_feat if i == 0 else agg_hidden
            self.gin_layers.append(GINLayer(in_dim, agg_hidden, device))

        # Fully connected layers for final classification
        self.fc1 = nn.Linear(agg_hidden, fc_hidden).to(device)
        self.fc2 = nn.Linear(fc_hidden, n_class).to(device)

    def forward(self, x, adj):
        # Move data to the correct device
        x = x.to(self.device)
        adj = adj.to(self.device)

        for i in range(self.n_layer):
            # Apply GIN layer with ReLU activation
            x = F.relu(self.gin_layers[i](x, adj))
            if i != self.n_layer - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply pooling operation (e.g., global average pooling)
        x = pooling_op(x, self.pool_type)  # Output should be of shape [1, agg_hidden]
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
