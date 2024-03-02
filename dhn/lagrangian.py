import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .types import *

class Lagrangian():
    
    @staticmethod
    def identity(x):
        """The Lagrangian whose activation function is simply the identity."""
        l =  0.5 * torch.pow(x, 2).sum()
        return l
    
    @staticmethod
    def repu(x: TENSOR, n):
        """Rectified Power Unit of degree `n`"""
        l = 1 / n * torch.pow(torch.clamp(x,0), n).sum() 
        return l

    @staticmethod
    def relu(x):
        """Rectified Linear Unit. Same as repu of degree 2"""
        return Lagrangian.repu(x, 2)