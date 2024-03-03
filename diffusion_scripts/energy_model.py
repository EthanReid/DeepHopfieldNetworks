import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence, Tuple
import dhn
from dhn.misc import TENSOR

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

class Energy_Model_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_block = Energy_Hop_T()
    
    def evolve(self, x: TENSOR, alpha: float = 1):
        dEdg, E = torch.func.grad_and_value(self.energy_block)(x)
        #x = x - alpha * dEdg
        return dEdg

    def forward(self, x: TENSOR):
        return self.evolve(x)

class Energy_Model_linked(nn.Module):
    def __init__(self,
                 in_dim: int,
                 multiplier: int,
                 bias: bool = False
                 ):
        super().__init__()
        self.energy_blocks_linked = Energy_Blocks_linked(in_dim, multiplier=multiplier, bias=bias)
    def evolve(self, x: TENSOR, alpha: float = 1):
        dEdg, E = torch.func.grad_and_value(self.energy_blocks_linked)(x)
        #x = x - alpha * dEdg
        return dEdg

    def forward(self, x: TENSOR):
        return self.evolve(x)

class Energy_Model_Conv(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernal_size: Tuple[int, int],
                ):
        super().__init__()
        self.energy_block = Energy_Block_Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernal_size=kernal_size
        )
    
    def evolve(self, x: TENSOR, alpha: float = 1):
        dEdg, E = torch.func.grad_and_value(self.energy_block)(x)
        #x = x - alpha * dEdg
        return dEdg

    def forward(self, x: TENSOR):
        return self.evolve(x)

class Energy_Model_Conv_Linked(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernal_size: Tuple[int, int],
                ):
        super().__init__()
        self.energy_block = Energy_Block_Conv_Linked(
            in_channels=in_channels,
            out_channels=out_channels,
            kernal_size=kernal_size
        )
    
    def evolve(self, x: TENSOR, alpha: float = 1):
        dEdg, E = torch.func.grad_and_value(self.energy_block)(x)
        #x = x - alpha * dEdg
        return dEdg

    def forward(self, x: TENSOR):
        return self.evolve(x)

class Energy_Blocks_linked(nn.Module):
    def __init__(self,
                 in_dim: int,
                 multiplier: int,
                 bias: bool = False
                 ):
        super().__init__()
        self.input = dhn.layers.Input()
        self.energy_block_1 = _Energy_Block(in_dim=in_dim, multiplier=multiplier, bias=bias)
        self.energy_block_2 = _Energy_Block(in_dim=in_dim*multiplier, multiplier=1, bias=bias)
        self.energy_block_3 = _Energy_Block(in_dim=in_dim*multiplier, multiplier=2, bias=bias)
        self.energy_block_4 = _Energy_Block(in_dim=in_dim*multiplier*2, multiplier=2, bias=bias)
        self.energy_block_5 = _Energy_Block(in_dim=in_dim*multiplier*2*2, multiplier=2, bias=bias)
        self.energy_block_6 = _Energy_Block(in_dim=in_dim*multiplier*2*2*2, multiplier=0.5, bias=bias)
        self.energy_block_7 = _Energy_Block(in_dim=in_dim*multiplier*2*2, multiplier=0.5, bias=bias)
        self.energy_block_8 = _Energy_Block(in_dim=int(in_dim*multiplier*2), multiplier=0.5, bias=bias)
        self.energy_block_9 = _Energy_Block(in_dim=int(in_dim*multiplier), multiplier=0.5, bias=bias)
        self.energy_block_10 = _Energy_Block(in_dim=int(in_dim*multiplier*0.5), multiplier=0.5, bias=bias)
        self.energy_block_11 = _Energy_Block(in_dim=int(in_dim*multiplier*0.5*0.5), multiplier=0.5, bias=bias)
        self.flat = dhn.layers.Generic(
            transformation=nn.Flatten(2),
            lagrangian=dhn.Lagrangian.identity,
            activation=nn.Identity()
        )

        
    def forward(self, x: TENSOR):
        #x, e_x = self.flat(x)
        g_1, e_g_1 = self.input(x)
        g_2, e_g_2 = self.energy_block_1(g_1)
        g_3, e_g_3 = self.energy_block_2(g_2)
        g_4, e_g_4 = self.energy_block_3(g_3)
        g_5, e_g_5 = self.energy_block_4(g_4)
        g_6, e_g_6 = self.energy_block_5(g_5)
        g_7, e_g_7 = self.energy_block_6(g_6)
        g_8, e_g_8 = self.energy_block_7(g_7)
        g_9, e_g_9 = self.energy_block_8(g_8)
        g_10, e_g_10 = self.energy_block_9(g_9)
        g_11, e_g_11 = self.energy_block_10(g_10)
        g_12, e_g_12 = self.energy_block_11(g_11)
        return (e_g_1 + e_g_2 + e_g_3 + e_g_4 + e_g_5 + e_g_6 + e_g_7 + e_g_8 + e_g_9 + e_g_10 + e_g_11 + e_g_12)

class Energy_Block_Conv_Linked(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super().__init__()
        self.input = dhn.layers.Input()
        self.conv_1 = _Energy_Block_Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernal_size=kernal_size
        )
        self.conv_2 = _Energy_Block_Conv(
            in_channels=out_channels,
            out_channels=out_channels*2,
            kernal_size=kernal_size
        )
        self.conv_3 = _Energy_Block_Conv(
            in_channels=out_channels*2,
            out_channels=out_channels*2*2,
            kernal_size=(4,4)
        )
        self.conv_4 = _Energy_Block_Conv(
            in_channels=out_channels*2*2,
            out_channels=out_channels*2,
            kernal_size=(2,2)
        )
        self.conv_5 = _Energy_Block_Conv(
            in_channels=out_channels*2,
            out_channels=1,
            kernal_size=(2,2)
        )
        #self.hop_t = Energy_Hop_T()
    
    def forward(self, x: TENSOR):
        g_1, e_g_1 = self.input(x)
        g_2, e_g_2 = self.conv_1(g_1)
        g_3, e_g_3 = self.conv_2(g_2)
        g_4, e_g_4 = self.conv_3(g_3)
        g_5, e_g_5 = self.conv_4(g_4)
        g_6, e_g_6 = self.conv_5(g_5)
        #g_5, e_g_5 = self.flat(g_4)#doesnt work
        #e_g_6 = self.linear_x(g_5)
       # e_g_5 = self.hop_t(g_4)
        return (e_g_1 + e_g_2 + e_g_3 + e_g_4 + e_g_5 + e_g_6)

class Energy_Hop_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = dhn.Input()
        self.energy_block_1 = _Energy_Block(in_dim=28, multiplier=2, bias=F)
        self.energy_block_1T = _Energy_Block(in_dim=28, multiplier=2, bias=F)
        self.energy_block_2 = _Energy_Block(in_dim=28*2, multiplier=2, bias=F)
        self.energy_block_2T = _Energy_Block(in_dim=28*2, multiplier=2, bias=F)
        self.energy_block_3 = _Energy_Block(in_dim=28*2*2, multiplier=0.5, bias=F)
        self.energy_block_3T = _Energy_Block(in_dim=28*2*2, multiplier=0.5, bias=F)
    
    def forward(self, x: TENSOR):
        x_t = dhn.transpose(x)
        x, e_g_0 = self.input(x)
        x_t, e_g_0t = self.input(x_t)
        g_1, e_g_1 = self.energy_block_1(x)
        g_1t, e_g_1t = self.energy_block_1T(x)
        g_2, e_g_2 = self.energy_block_2(g_1)
        g_2t, e_g_2t = self.energy_block_2T(g_1t)
        g_3, e_g_3 = self.energy_block_3(g_2)
        g_3t, e_g_3t = self.energy_block_3T(g_2t)
        return (e_g_0 + e_g_0t + e_g_1 + e_g_1t + e_g_2 + e_g_2t + e_g_3 + e_g_3t)



class Energy_Block(nn.Module):
    def __init__(self,
                    in_dim: int,
                    multiplier: int,
                    bias: bool = False
                    ):
        super().__init__()
        self.input = dhn.layers.Input()
        self.flat = dhn.layers.Generic(
            transformation=nn.Flatten(2),
            lagrangian=dhn.Lagrangian.identity,
            activation=nn.Identity()
        )
        self.hop = dhn.layers.Linear(
            in_dim=in_dim, 
            multiplier=multiplier,
            bias=bias,
            lagrangian=dhn.Lagrangian.relu,
            activation=nn.ReLU()
        )
    
    def forward(self, x: TENSOR):
        #x, e_x = self.flat(x)
        g_1, e_g_1 = self.input(x)
        _, e_hop = self.hop(g_1)
        return (e_hop + e_g_1)
    

class Energy_Block_Conv(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernal_size: Tuple[int, int],
                stride: int=1,
                padding = "same",
                mean = 0.5,
                std = 0.01,
                lagrangian: Callable = None,
                activation: Callable = None):
        super().__init__()
        self.input = dhn.layers.Input()
        self.conv = dhn.layers.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernal_size=kernal_size,
            stride=stride,
            padding=padding,
            mean=mean,
            std=std,
            lagrangian=dhn.Lagrangian.relu,
            activation=nn.ReLU()
        )
    
    def forward(self, x: TENSOR):
        g_1, e_g_1 = self.input(x)
        _, e_conv = self.conv(g_1)
        return (e_conv + e_g_1)


class _Energy_Block_Conv(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernal_size: Tuple[int, int],
                stride: int=1,
                padding = "same",
                mean = 0.5,
                std = 0.01,
                lagrangian: Callable = None,
                activation: Callable = None):
        super().__init__()
        self.input = dhn.layers.Input()
        self.conv = dhn.layers.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernal_size=kernal_size,
            stride=stride,
            padding=padding,
            mean=mean,
            std=std,
            lagrangian=dhn.Lagrangian.relu,
            activation=nn.ReLU()
        )
    
    def forward(self, g_1: TENSOR):
        g_2, e_conv = self.conv(g_1)
        return g_2, e_conv

class _Energy_Block(nn.Module):
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
    
    def forward(self, g_1: TENSOR):
        g_2, e_hop = self.hop(g_1)
        return g_2, e_hop
