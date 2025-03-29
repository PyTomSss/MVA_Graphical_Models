import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer, GraphAttentionLayer, GraphDenseNetLayer
from pooling import pooling_op


class GCN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GCN, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.pool_type = pool_type
        
        # Graph convolution layer
        self.graph_convolution_layers = []
        for i in range(n_layer):
           if i == 0:
             self.graph_convolution_layers.append(GraphConvolutionLayer(n_feat, agg_hidden, device))
           else:
             self.graph_convolution_layers.append(GraphConvolutionLayer(agg_hidden, agg_hidden, device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
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
    

class GAT(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, pool_type, device):
        super(GAT, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.pool_type = pool_type
        
        # Graph attention layer
        self.graph_attention_layers = []
        for i in range(self.n_layer):
          self.graph_attention_layers.append(GraphAttentionLayer(n_feat, agg_hidden, dropout, device))
                    
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden*n_layer, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
        
    def forward(self, data):
        x, adj = data[:2]
        
        # Dropout        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph attention layer
        x = torch.cat([F.relu(att(x, adj)) for att in self.graph_attention_layers], dim=2)

        # pool_type
        x = pooling_op(x, self.pool_type)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        
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