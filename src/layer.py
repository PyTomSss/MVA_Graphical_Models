import math
import torch
import torch.nn as nn
import torch.nn.functional as F

### IN THIS FILE, WE DEFINE LAYERS THAT WILL BE USED IN MODELS.PY TO DEFINE GNNs

## 1. Graph Convolution Layer

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialisation des poids
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    
    def forward(self, x, adj):
        x = x.reshape(-1, x.size(1))
        x = torch.mm(x, self.weight)
        x = x.reshape(adj.size()[0], adj.size()[1], self.weight.size()[-1])
        output = torch.bmm(adj, x)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

## 2. Graph Attention Layer


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # Number of input features
        self.out_features = out_features  # Number of output features
        self.dropout = dropout  # Dropout rate for attention weights
        
        # Learnable weight matrix for feature transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        # Learnable weight vector for computing attention coefficients
        self.weight2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(device)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the learnable parameters using a uniform distribution.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
                        
    def forward(self, x, adj):
        """
        Forward pass for the Graph Attention Layer.

        Args:
            x (torch.Tensor): Input feature matrix of shape (batch_size, num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output feature matrix after attention mechanism.
        """
        batch_size = x.size(0)
        node_count = x.size(1)
        
        # Reshape input and apply linear transformation
        x = x.reshape(batch_size * node_count, x.size(2))
        x = torch.mm(x, self.weight)  # Linear transformation
        x = x.reshape(batch_size, node_count, self.weight.size(-1))
        
        # Compute attention scores
        attention_input = torch.cat([
            x.repeat(1, 1, node_count).view(batch_size, node_count * node_count, -1),  # Repeating x along nodes
            x.repeat(1, node_count, 1)  # Repeating x along feature dimension
        ], dim=2).view(batch_size, node_count, -1, 2 * self.out_features)  # Concatenation for attention
        
        e = F.relu(torch.matmul(attention_input, self.weight2).squeeze(3))  # Compute raw attention scores
        zero_vec = -9e15 * torch.ones_like(e)  # Masking for non-existent edges (very negative values)
        attention = torch.where(adj > 0, e, zero_vec)  # Apply adjacency mask
        attention = F.softmax(attention, dim=2)  # Normalize attention scores
        attention = F.dropout(attention, self.dropout, training=self.training)  # Apply dropout to attention
        
        # Compute new node features using weighted sum
        x = torch.bmm(attention, x)  # Matrix multiplication of attention weights with transformed features
        
        return x


## 3. Graph Dense Net


class GraphDenseNetLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphDenseNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(in_features, out_features, device)
        self.graph_convolution_layer2 = GraphConvolutionLayer(in_features + out_features, out_features, device)

        # Pooling
        self.pooling = nn.Parameter(torch.FloatTensor(out_features * 4, out_features)).to(device)
        self.pooling_ration = 0.8
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.pooling.size(1))
        self.pooling.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, pooling):
        
        # Pooling
        if pooling:
            B, N, C = x.size()
            x = x.reshape(B * N, C)
            x = torch.mm(x, self.pooling).reshape(B, N, self.out_features)
            
        # Graph convolution layer
        x1 = self.graph_convolution_layer(x, adj)
        
        # Concat
        concat1 = torch.cat((x, x1), 2)
        
        # Graph convolution layer        
        x2 = self.graph_convolution_layer2(concat1, adj)
        
        # Concat
        concat2 = torch.cat((x, concat1, x2), 2)
        
        return concat2