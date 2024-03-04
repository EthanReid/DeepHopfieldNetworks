import torch
from typing import TypeAlias
from einops import rearrange
TENSOR: TypeAlias = torch.tensor
from inspect import isfunction

def transpose(x: TENSOR) -> TENSOR:
    """Transposes the last two dimensions of a tensor"""
    return rearrange(x, "... h w -> ... w h")

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Counter():
    def __init__(self, value):
        self.value = value
    def add(self, x):
        self.value += x
    def subtract(self, x):
        self.value -= x