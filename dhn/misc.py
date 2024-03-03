import torch
from typing import TypeAlias
from einops import rearrange
TENSOR: TypeAlias = torch.tensor

def transpose(x: TENSOR) -> TENSOR:
    """Transposes the last two dimensions of a tensor"""
    return rearrange(x, "... h w -> ... w h")
