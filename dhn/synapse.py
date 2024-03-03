import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .misc import *
from einops import rearrange

class Synapse(nn.Module):
    def __init__(self):
        super().__init__()
    
    #trying out inner product, the outer product works very well, but not sure if its right
    def energy(self, g_2: TENSOR, x_2: TENSOR) -> TENSOR:
        #rewrtie to just return e, this is for testing
        g_2 = transpose(g_2) #create a generic transpose function
        e_mm = torch.matmul(x_2, g_2)
        #e_mm = torch.matmul(g_2, x_2)
        e_sum = e_mm.sum()
        return -e_sum
    
    def forward(self, g_2: TENSOR, x_2: TENSOR) -> TENSOR:
        return self.energy(g_2, x_2)