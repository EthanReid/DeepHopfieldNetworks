import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .neuron import Neuron
from .synapse import Synapse
from .lagrangian import *
from .misc import *

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

class Generic(nn.Module):
    def __init__(self, transformation, lagrangian: Callable = None, activation: Callable = None):
        super().__init__()
        self.neuron = Neuron(lagrangian=lagrangian, activation=activation)
        self.transformation = transformation
        self.synapse = Synapse()
    
    def forward(self, g_1: TENSOR):
        '''accepts g_1 (activation vector of x_1, assumes energy has already been recorded)
        return g_2, and the energy total of g_2 and synapse'''
        x_2 = self.transformation(g_1)
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

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernal_size: Tuple[int, int],
                 stride: int,
                 padding = "same",
                 mean = 0.5,
                 std = 0.01,
                 lagrangian: Callable = None,
                 activation: Callable = None):
        super().__init__()
        if padding != "same":
            padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernal_size,
                              stride=stride,
                              padding=padding)
        #nn.init.normal_(self.conv.weight, mean=mean, std=std)
        #nn.init.constant_(self.conv.bias, 0)
        self.neuron = Neuron(lagrangian=lagrangian, activation=activation)
        self.synapse = Synapse()
    
    def forward(self, g_1: TENSOR):
        x_2 = self.conv(g_1)
        g_2, e_g_2 = self.neuron(x_2)
        e_s = self.synapse(g_2, x_2)
        return g_2, (e_g_2 + e_s)