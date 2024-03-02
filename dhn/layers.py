import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .neuron import Neuron
from .synapse import Synapse
from .lagrangian import *
from .types import *

class Linear(nn.Module):
    def __init__(self, in_dim, multiplier, bias= False, lagrangian: Callable = None, activation: Callable = None):
        super().__init__()
        self.neuron = Neuron(lagrangian=lagrangian, activation=activation)
        self.linear = nn.Linear(in_dim, int(in_dim*multiplier),bias=bias)
        self.synapse = Synapse()
    
    def forward(self, g_1: TENSOR):
        '''accepts g_1 (activation vector of x_1, assumes energy has already been recorded)
        return g_2, and the energy total of g_2 and synapse'''
        x_2 = self.linear(g_1)
        g_2, e_g_2 = self.neuron(x_2)
        e_s = self.synapse(g_2, x_2)
        return g_2, (e_g_2 + e_s)

class Input(nn.Module):
    '''
    Intended to be the first layer of the network. Takes x_1 and returns g_1, and e_g_1.
    Acts as a wrapper for the identity lagrangian
    '''
    def _init__(self):
        super().__init__()
    
    def forward(self, x: TENSOR):
        e_g = Lagrangian.identity(x)
        return x, e_g