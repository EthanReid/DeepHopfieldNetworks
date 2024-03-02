import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .types import *
from einops import rearrange

class Synapse(nn.Module):
    def __init__(self):
        super().__init__()
    
    def energy(self, g_2, x_2):
        #rewrtie to just return e, this is for testing
        g_2 = rearrange(g_2, "b c h w -> b c w h") #create a generic transpose function
        e_mm = torch.matmul(g_2, x_2)
        e_sum = e_mm.sum()
        return e_sum
    
    def forward(self, g_2, x_2):
        return self.energy(g_2, x_2)