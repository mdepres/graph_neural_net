import torch
import torch.nn as nn

class ColoringModel(nn.module):
    """ A simple network to learn coloring based on prelearned node embeddings"""
    def __init__(self, n_vertices, embed_dim, hidden_dim = 32):
        self.mlp1 = nn.Linear(embed_dim,hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU
    
    def forward(self, emb_in):
        out = self.mlp1(emb_in)
        out = self.relu(out)
        out = self.mlp2(out)
        out = self.relu(out)
        return out
