import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

from catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "IED":
        return IED(config.tokens, config.graph.p_m, config.graph.type, config.graph.loss_type)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens, config.graph.p_m, config.graph.type, config.graph.loss_type)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass

class Absorbing(Graph):
    def __init__(self, dim, p_m, graph_type, loss_type):
        self._dim = dim
        self.p_m = p_m
        self.graph_type = graph_type
        self.loss_type = loss_type

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True


    # def sample_transition(self, i, t):
    #     k_t=t
    #     move_chance = 1-k_t
    #     move_indices = torch.rand(*i.shape, device=i.device) < move_chance
    #     i_pert = torch.where(move_indices, self.dim - 1, i)
    #     return i_pert
    
    def sample_transition(self, sourcebatch, databatch, t):
        k_t=t
        move_chance = 1-k_t
        move_indices = torch.rand(*databatch.shape, device=databatch.device) < move_chance
        pert_batch = sourcebatch*move_indices+databatch*(~move_indices)
        return pert_batch
    
    def sample_limit(self, batch_dims):
        B, L = batch_dims
        return (self.dim - 1) * torch.ones(B, L, dtype=torch.int64)
    

class IED(Graph):
    def __init__(self, dim, p_m, graph_type, loss_type):
        self._dim = dim
        self.p_m = p_m
        self.graph_type = graph_type
        self.loss_type = loss_type

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def sample_transition(self, sourcebatch, databatch, t):
        k_t=t
        move_chance = 1-k_t
        move_indices = torch.rand(*databatch.shape, device=databatch.device) < move_chance
        pert_batch = sourcebatch*move_indices+databatch*(~move_indices)
        return pert_batch
    
    def sample_limit(self, batch_dims):
        B, L = batch_dims
        token_distribution = torch.load('td.pth').squeeze()[:8]
        tokens = torch.multinomial(token_distribution, B*L, replacement=True)
        batch = tokens.reshape(B, L).to(torch.int64)
        return batch