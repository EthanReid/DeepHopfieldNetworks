import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence
import dhn
from dhn.types import TENSOR

class Energy_Model(nn.Module):
    def __init__(self,
                 in_dim: int,
                 multiplier: int,
                 bias: bool = False
                 ):
        super().__init__()
        self.energy_block = Energy_Block(in_dim=in_dim, multiplier=multiplier, bias=bias)
    
    def evolve(self, x: TENSOR, alpha: float = 1):
        dEdg, E = torch.func.grad_and_value(self.energy_block)(x)
        #x = x - alpha * dEdg
        return dEdg

    def forward(self, x: TENSOR):
        return self.evolve(x)
class Energy_Block(nn.Module):
    def __init__(self,
                    in_dim: int,
                    multiplier: int,
                    bias: bool = False
                    ):
        super().__init__()
        self.input = dhn.layers.Input()
        self.hop = dhn.layers.Linear(
            in_dim=in_dim, 
            multiplier=multiplier,
            bias=bias,
            lagrangian=dhn.Lagrangian.relu,
            activation=nn.ReLU()
        )
    
    def forward(self, x: TENSOR):
        g_1, e_g_1 = self.input(x)
        _, e_hop = self.hop(g_1)
        return (e_hop + e_g_1)
