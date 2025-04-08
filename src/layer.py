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
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass for the Graph Convolutional Layer.

        Args:
            x (torch.Tensor): Input feature matrix of shape (batch_size, num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output feature matrix after attention mechanism.
        """
        support = torch.mm(x, self.weight)         # [N, out_features]
        output = torch.mm(adj, support)            # [N, out_features]
        if self.bias is not None:
            output += self.bias
        return output

## 2. Graph Attention Layer

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device, alpha=0.2, bias=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.device = device

        # Linear transformation
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1)).to(device)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, device=device))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: [N, in_features]
        adj: [N, N]
        """
        N = x.size(0)

        h = torch.mm(x, self.W)  # [N, out_features]
        # Prepare attention mechanism input (all pairwise combinations)
        a_input = torch.cat([
            h.repeat(1, N).view(N * N, -1),        # h_i
            h.repeat(N, 1)                          # h_j
        ], dim=1).view(N, N, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        # Masked attention: remove non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # Normalize
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)  # [N, out_features]

        if self.bias is not None:
            h_prime += self.bias

        return h_prime


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
    

# 4. GIN 

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, device, eps=0.0):
        super(GINLayer, self).__init__()
        self.device = device
        self.eps = nn.Parameter(torch.Tensor([eps]))  # Learnable epsilon
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        ).to(device)

    def forward(self, x, adj):
        # Sum aggregation
        agg = torch.matmul(adj, x)  # [N, F]
        out = (1 + self.eps) * x + agg
        out = self.mlp(out)
        return out
