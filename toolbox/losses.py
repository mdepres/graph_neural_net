import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from toolbox.utils import get_device


class triplet_loss(nn.Module):
    def __init__(self, loss_reduction='mean', loss=nn.CrossEntropyLoss(reduction='sum')):
        super(triplet_loss, self).__init__()
        self.loss = loss
        if loss_reduction == 'mean':
            self.increments = lambda new_loss, n_vertices : (new_loss, n_vertices)
        elif loss_reduction == 'mean_of_mean':
            self.increments = lambda new_loss, n_vertices : (new_loss/n_vertices, 1)
        else:
            raise ValueError('Unknown loss_reduction parameters {}'.format(loss_reduction))

# !!! to be checked: only working with graphs same size ?!!!
    def forward(self, raw_scores):
        """
        raw_scores is the output of siamese network (bs,n_vertices,n_vertices)
        """
        device = get_device(raw_scores)
        loss = 0
        total = 0
        for out in raw_scores:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss/total

class coloring_loss(nn.Module):
    """ Returns a loss for the coloring problem """
    def __init__(self):
        super(coloring_loss, self).__init__()
    
    def forward(self, W, pred, tgt, eps=1e-2):
        mark = torch.zeros((1),requires_grad=True)
        for b in range(W.shape[0]):
            for i in range(W.shape[1]):
                for j in range(W.shape[2]):
                    if W[b][1][i][j].item()==1 and torch.abs(pred[b][i]-pred[b][j])<eps:
                        mock = torch.ones((1))
                        mark = torch.add(mark,mock)
        return mark
