import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .misc import *
from einops import rearrange

class Neuron(nn.Module):
    def __init__(self, lagrangian: Callable, activation: Callable = None): 
        super().__init__()
        self._lagrangian = lagrangian
        self._activation = activation if activation else F.grad(self._lagrangian)
    
    def activation(self, x: TENSOR) -> TENSOR:
        return self._activation(x)
    
    #there might be an error in my matmul, matmul(g^t,x) was giving the outer product.
    def energy(self, g: TENSOR, x: TENSOR) -> TENSOR: #not sure what the type would be, scalar or a vector?
        #print("g type: {}\n x type: {}".format(g.type(), x.type()))
        g = transpose(g) #do I need a transpose, should this happen before, assume [B, C, (H*W)] or [B, (C*H*W)]. RN its [B, C, H, W]
        l = self._lagrangian(x)
        e = torch.matmul(x, g).sum() - l
        #e = torch.matmul(g, x).sum() - l
        return e
    
    def forward(self, x: TENSOR): #idk return type as I dont know return of energy
        g = self.activation(x)
        e_g =self.energy(g, x)
        return g, e_g

#add other neuron types
