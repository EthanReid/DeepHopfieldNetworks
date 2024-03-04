import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence, Tuple
import dhn
from dhn.misc import TENSOR

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, init_shape=None):
        super().__init__()
        self.proj = dhn.layers.Conv2d(in_channels=dim, out_channels=dim_out, kernal_size=(3,3))
        #self.norm = dhn.neuron.Neuron(lagrangian=dhn.Lagrangian.layernorm, activation=nn.LayerNorm(init_shape))
        self.act = dhn.neuron.Neuron(lagrangian=dhn.Lagrangian.relu, activation=nn.ReLU())

    def forward(self, x, scale_shift=None):
        x, e_proj = self.proj(x)
        #x, e_norm = self.norm(x)
        x, e_act = self.act(x)
        return x, (e_proj+e_act)

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, init_shape=None):
        super().__init__()
        self.init_shape = None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = dhn.layers.Conv2d(in_channels=dim, out_channels=dim_out, kernal_size=(1,1), padding=0) if dim!=dim_out else dhn.layers.Input()
        self.add = dhn.layers.Add(lagrangian=dhn.Lagrangian.identity, activation=nn.Identity())
    def forward(self, x):
        h, e_b_1 = self.block1(x)
        h, e_b_2 = self.block2(h)
        x_res_conv, e_res_conv = self.res_conv(x)
        x_h, e_add = self.add(h, x_res_conv)
        return x_h, (e_b_1+e_b_2+e_res_conv+e_add)
