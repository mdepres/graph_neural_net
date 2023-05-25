import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    """ A simple network to use prelearned node embeddings to learn graph problems that can be seen as "classification" tasks
    Exemples : Graph coloring for fixed k or Minimum bisection (k=2) """
    def __init__(self, n_vertices, embed_dim, k, hidden_dim = 64):
        super(Classifier_model, self).__init__()
        self.mlp1 = nn.Linear(embed_dim,hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, k)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, emb_in):
        out = self.mlp1(emb_in)
        out = self.relu(out)
        out = self.mlp2(out)
        out = self.softmax(out)
        return out
