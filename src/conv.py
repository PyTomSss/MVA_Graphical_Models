import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features, conv_type="gcn"):
        super(GNNLayer, self).__init__()
        self.conv_type = conv_type.lower()

        if conv_type == "gcn":
            # GCN layer
            self.weight = nn.Parameter(torch.randn(in_features, out_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.activation = F.relu
        elif conv_type == "gat":
            # GAT layer (simplified version)
            self.attention_weight = nn.Parameter(torch.randn(in_features, out_features))
            self.attention = nn.Parameter(torch.randn(2*out_features, 1))
            self.activation = F.elu
        elif conv_type == "graphsage":
            # GraphSAGE layer
            self.weight = nn.Parameter(torch.randn(in_features, out_features))
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown conv_type {conv_type}")

    def forward(self, h_i, h_neighbors, adj_matrix=None):
        if self.conv_type == "gcn":
            # GCN update rule: h_i' = A * h_i + W * h_neighbors
            h_neighbors_mean = torch.mean(h_neighbors, dim=0)
            h_i_prime = F.relu(torch.matmul(h_i, self.weight) + self.bias + h_neighbors_mean)
        
        elif self.conv_type == "gat":
            # GAT: Attention mechanism between nodes
            h_i_prime = torch.matmul(h_i, self.attention_weight)
            attention_scores = torch.matmul(torch.cat([h_i_prime, h_neighbors], dim=0), self.attention)
            attention_scores = F.softmax(attention_scores, dim=0)
            h_i_prime = torch.sum(attention_scores * h_neighbors, dim=0)

        elif self.conv_type == "graphsage":
            # GraphSAGE: h_i' = W * h_i + mean(W * h_neighbors)
            h_neighbors_mean = torch.mean(torch.matmul(h_neighbors, self.weight), dim=0)
            h_i_prime = F.relu(torch.matmul(h_i, self.weight) + h_neighbors_mean)
        
        return h_i_prime
    
class BasicGraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W2 = Parameter(torch.rand(
            (in_channels, out_channels), dtype=torch.float32))
        self.W1 = Parameter(torch.rand(
            (in_channels, out_channels), dtype=torch.float32))
        self.bias = Parameter(torch.zeros(
                out_channels, dtype=torch.float32))
    def forward(self, X, A):
        potential_msgs = torch.mm(X, self.W2)
        propagated_msgs = torch.mm(A, potential_msgs)
        root_update = torch.mm(X, self.W1)
        output = propagated_msgs + root_update + self.bias
        return output
    
