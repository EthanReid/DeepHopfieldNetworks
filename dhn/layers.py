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
                 stride: int = 1,
                 padding = "same",
                 mean = 0.5,
                 std = 0.01,
                 lagrangian: Callable = Lagrangian.identity,
                 activation: Callable = nn.Identity()):
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

class MaxPool2d(nn.Module):
    def __init__(self,
                 kernal_size: Tuple[int,int],
                 stride: int,
                 padding: int,
                 lagrangian: Callable = Lagrangian.identity,
                 activation: Callable = nn.Identity()):
        super().__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernal_size,
            stride=stride,
            padding=padding
        )
        self.neuron = Neuron(lagrangian=lagrangian, activation=activation)
        self.synapse = Synapse()
    
    def forward(self, g_1: TENSOR):
        x_2 = self.pool(g_1)
        g_2, e_g_2 = self.neuron(x_2)
        e_s = self.synapse(g_2, x_2)
        return g_2, (e_g_2 + e_s)

class Add(nn.Module):
    def __init__(self,
                 lagrangian: Callable = None,
                 activation: Callable = None):
        super().__init__()
        self.neuron = Neuron(lagrangian=lagrangian, activation=activation)
        self.synapse = Synapse()
    
    def forward(self, g_1: TENSOR, g_2: TENSOR):
        x_3 = g_1 + g_2
        g_3, e_g_3 = self.neuron(x_3)
        e_s = self.synapse(g_3, x_3)
        return g_3, (e_g_3 + e_s)

class ConvAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = Conv2d(dim, hidden_dim * 3, kernal_size=(1,1))
        self.to_out = Conv2d(hidden_dim, dim, kernal_size=(1,1))
        self.softmax = Neuron(lagrangian=Lagrangian.softmax, activation=nn.Softmax(-1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv, e_qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn, e_attn = self.softmax(sim)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out, e_out = self.to_out(out)
        return out, e_attn+e_out+e_qkv


def MakeLayer(transformation: Callable, lagrangian: Callable = None, activation: Callable = None):
    return Generic(transformation, lagrangian, activation)

#add a make layer func that can reuse the generic