import torch
from collections import OrderedDict


class MLP(torch.nn.Module):
    def __init__(self, L, in_dim, h_dim, out_dim):
        super().__init__()
        layers = []
        for i in range(L):
            if i == 0:
                layers.append(torch.nn.Linear(in_dim, h_dim))
                layers.append(torch.nn.ReLU())
            elif i == L-1:
                layers.append(torch.nn.Linear(h_dim, out_dim))
            else:
                layers.append(torch.nn.Linear(h_dim, h_dim))
                layers.append(torch.nn.ReLU())
        layers = OrderedDict([(str(i), layer) for i, layer in enumerate(layers)])
        self.layers = torch.nn.Sequential(layers)
        
    def forward(self, x):
        out = self.layers(x)
        return out