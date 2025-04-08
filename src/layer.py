# ==============================================================================
# File: layer.py
#
# Description:
# In this file, we define several layers used to construct Graph Neural Networks (GNNs).
# The layers implemented here are:
# 1. Graph Convolution Layer (GraphConv)
# 2. Graph Attention Layer (GAT)
# 3. Graph Isomorphism Network Layer (GIN)
# These layers will be utilized in models defined in the "models.py" file.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

## 1. Graph Convolution Layer
    
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        """
        Initialize the Graph Convolution Layer.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            device (torch.device): The device (CPU or GPU) where the layer's parameters will be stored.
            bias (bool): Whether to include a bias term (default: True).
        """
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight initialization for the convolutional layer
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        
        # If bias is True, initialize the bias term
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the weights and biases using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Perform a forward pass through the Graph Convolution Layer.

        Args:
            x (torch.Tensor): Input feature matrix of shape (batch_size, num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output feature matrix after applying the graph convolution.
        """
        support = torch.mm(x, self.weight)  # [N, out_features]
        output = torch.mm(adj, support)    # [N, out_features]
        
        if self.bias is not None:
            output += self.bias
        
        return output

## 2. Graph Attention Layer

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device, alpha=0.2, bias=True):
        """
        Initialize the Graph Attention Layer (GAT).

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            dropout (float): Dropout probability to be applied to the attention scores.
            device (torch.device): The device (CPU or GPU) where the layer's parameters will be stored.
            alpha (float): The negative slope used in LeakyReLU activation (default: 0.2).
            bias (bool): Whether to include a bias term (default: True).
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.device = device

        # Linear transformation matrix for node features
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        
        # Attention mechanism: learnable parameter for edge importance
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1)).to(device)

        # Bias term (optional)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, device=device))
        else:
            self.register_parameter('bias', None)

        # LeakyReLU activation for attention scores
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the weights, attention parameters, and bias term using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Perform a forward pass through the Graph Attention Layer.

        Args:
            x (torch.Tensor): Input feature matrix of shape (num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output feature matrix after applying attention mechanism.
        """
        N = x.size(0)

        # Apply the linear transformation to the node features
        h = torch.mm(x, self.W)  # [N, out_features]

        # Prepare attention mechanism input (all pairwise combinations)
        a_input = torch.cat([
            h.repeat(1, N).view(N * N, -1),  # h_i
            h.repeat(N, 1)                    # h_j
        ], dim=1).view(N, N, 2 * self.out_features)

        # Compute attention scores using the learnable attention parameter
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        # Masking the attention for non-edges (no connection)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax to normalize attention scores
        attention = F.softmax(attention, dim=1)
        
        # Apply dropout to the attention scores
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Compute the new feature values by weighted sum of neighbors
        h_prime = torch.matmul(attention, h)  # [N, out_features]

        if self.bias is not None:
            h_prime += self.bias

        return h_prime  

## 3. Graph Isomorphism Network Layer (GIN)

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, device, eps=0.0):
        """
        Initialize the Graph Isomorphism Network (GIN) Layer.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            device (torch.device): The device (CPU or GPU) where the layer's parameters will be stored.
            eps (float): The learnable epsilon parameter for the GIN layer (default: 0.0).
        """
        super(GINLayer, self).__init__()
        self.device = device
        self.eps = nn.Parameter(torch.Tensor([eps]))  # Learnable epsilon term
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        ).to(device)

    def forward(self, x, adj):
        """
        Perform a forward pass through the GIN Layer.

        Args:
            x (torch.Tensor): Input feature matrix of shape (num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output feature matrix after applying the sum aggregation and MLP.
        """
        # Sum aggregation over the neighbors
        agg = torch.matmul(adj, x)  # [N, F]
        
        # Apply epsilon-based transformation
        out = (1 + self.eps) * x + agg
        
        # Apply the Multi-Layer Perceptron (MLP)
        out = self.mlp(out)
        
        return out
