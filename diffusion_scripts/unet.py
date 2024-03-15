import torch
import torch.nn as nn
import torch.nn.functional as F
import dhn
from dhn.misc import TENSOR
from functools import partial
from .resnet import *
from einops.layers.torch import Rearrange
from .energy_model import Energy_Hop_T
#nNote: Missing position embedding, groupnorm (in resnet), linear attention, and prenorm

class Unet_Model(nn.Module):
    def __init__(self,
                 dim,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 self_condition=False,
                 resnet_block_groups=4,
                 init_shape = None):
        super().__init__()
        self.unet = Unet(dim=dim,
                         init_dim=init_dim,
                         out_dim=out_dim,
                         dim_mults=dim_mults,
                         channels=channels,
                         self_condition=self_condition,
                         resnet_block_groups=resnet_block_groups,
                         init_shape=init_shape)
    
    def forward(self, x: TENSOR):
        dEdg, E = torch.func.grad_and_value(self.unet)(x)
        return dEdg

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
        init_shape = None
    ):
        super().__init__()
        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        self.init_shape = init_shape

        init_dim = dhn.misc.default(init_dim, dim)
        self.init_conv = dhn.layers.Conv2d(in_channels=input_channels, out_channels=init_dim, kernal_size=(1,1),padding=0)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.input = dhn.layers.Input()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        #dhn.layers.ConvAttention(dim_in),
                        #nn.GroupNorm(1,dim_in),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else dhn.Conv2d(dim_in, dim_out, kernal_size=(3,3)), #padding changed from 1
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
       # self.mid_attn = dhn.ConvAttention(mid_dim)
       # self.mid_norm = nn.GroupNorm(1,mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out),
                        block_klass(dim_out + dim_in, dim_out),
                       # dhn.ConvAttention(dim_out),
                       # nn.GroupNorm(1,dim_out),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else dhn.Conv2d(dim_out, dim_in, kernal_size=(3,3)),#padding changed from 1 to default, same
                    ]
                )
            )

        self.out_dim = dhn.misc.default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = dhn.Conv2d(dim, self.out_dim, kernal_size=(1,1))
        self.add = dhn.layers.Add()
        #self.hop_t = Energy_Hop_T()
    def forward(self, x, x_self_cond=None):
        #e_hop = self.hop_t(x)
        x, e_x = self.input(x)
        e_count = dhn.Counter(e_x)

        if self.self_condition:
            x_self_cond = dhn.misc.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x, e_x = self.init_conv(x)
        e_count.add(e_x)
        r = x.clone()

        h = []

        for block1, block2,downsample in self.downs: # attn, norm, 
            x, e_x = block1(x)
            e_count.add(e_x)
            h.append(x)

            x, e_x = block2(x)
            e_count.add(e_x)
           # att, e_attn = attn(x) #this is not linear attn, and doesnt not add the residual
           # att = norm(att)
            #x, e_x = self.add(att, x)
           # e_count.add(e_attn)
            h.append(x)

            x, e_x = downsample(x)
            e_count.add(e_x)

        x, e_x = self.mid_block1(x)
        e_count.add(e_x)
        #attn, e_attn = self.mid_attn(x) #this is not linear attn, and doesnt not add the residual
        #attn = self.mid_norm(attn)
        #x, e_x = self.add(attn, x)
        #e_count.add(e_attn)
        x, e_x = self.mid_block2(x)
        e_count.add(e_x)

        for block1, block2, upsample in self.ups: # attn, norm,
            x = torch.cat((x, h.pop()), dim=1)
            x, e_x = block1(x)
            e_count.add(e_x)

            x = torch.cat((x, h.pop()), dim=1)
            x, e_x = block2(x)
            e_count.add(e_x)
           # att, e_attn = attn(x) #this is not linear attn, and doesnt not add the residual
           # att = norm(att)
            #x, e_x = self.add(attn, x)
           # e_count.add(e_attn)

            x, e_x = upsample(x)
            e_count.add(e_x)

        x = torch.cat((x, r), dim=1)

        x, e_x = self.final_res_block(x)
        e_count.add(e_x)
        
        x, e_x = self.final_conv(x)
        e_count.add(e_x)
        return e_count.value

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.conv = dhn.Conv2d(dim*4, dhn.default(dim_out, dim), kernal_size=(1,1), padding=0)
    
    def forward(self, x:TENSOR):
        x = rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        x, e_x = self.conv(x)
        return x, e_x

class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")#dont have a energy implimentation at the moment
        self.conv = dhn.Conv2d(dim, dhn.default(dim_out, dim), kernal_size=(3,3))

    def forward(self, x:TENSOR):
        x = self.upsample(x)
        x, e_x = self.conv(x)
        return x, e_x
    
#not energy based
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)