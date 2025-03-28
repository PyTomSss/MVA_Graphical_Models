import torch
import torch.nn as nn 

class PoolingLayer(nn.Module):
    def __init__(self, pool_type="mean"):
        super(PoolingLayer, self).__init__()
        self.pool_type = pool_type

    def forward(self, h):
        if self.pool_type == "mean":
            return torch.mean(h, dim=0)
        elif self.pool_type == "max":
            return torch.max(h, dim=0)[0]
        elif self.pool_type == "sum":
            return torch.sum(h, dim=0)
        else:
            raise ValueError(f"Unknown pool_type {self.pool_type}")
