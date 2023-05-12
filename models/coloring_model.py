import torch
import torch.nn as nn

class ColoringModel(nn.Module):
    """ A simple network to learn coloring based on prelearned node embeddings"""
    def __init__(self, n_vertices, embed_dim, k, hidden_dim = 32):
        super(ColoringModel, self).__init__()
        self.mlp1 = nn.Linear(embed_dim,hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, k)
        self.relu = nn.ReLU()

    
    def forward(self, emb_in):
        out = self.mlp1(emb_in)
        out = self.relu(out)
        out = self.mlp2(out)
        out = self.relu(out)
        return out
