import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Dict, List, Optional
import torch.func as F
from .misc import *

class Lagrangian():
    
    @staticmethod
    def identity(x: TENSOR) -> TENSOR:
        """The Lagrangian whose activation function is simply the identity."""
        l =  0.5 * torch.pow(x, 2).sum()
        return l
    
    @staticmethod
    def repu(x: TENSOR, n: int) -> TENSOR:
        """Rectified Power Unit of degree `n`"""
        l = 1 / n * torch.pow(torch.clamp(x,0), n).sum() 
        return l

    @staticmethod
    def relu(x: TENSOR) -> TENSOR:
        """Rectified Linear Unit. Same as repu of degree 2"""
        return Lagrangian.repu(x, 2)
    
    @staticmethod
    def sigmoid(x: TENSOR, beta=1.0, scale=1.0) -> TENSOR:
        """The lagrangian of a sigmoid that we can define custom JVPs of"""
        return (scale / beta * torch.log(torch.exp(beta * x) + 1)).sum()

    @staticmethod
    def softmax(x: TENSOR, beta=1.0, dim=-1, keepdim=True):
        """The lagrangian of the softmax -- the logsumexp"""
        return (1/beta * torch.logsumexp(beta * x, dim=dim, keepdim=keepdim)).sum()
    
    @staticmethod
    def layernorm(x: TENSOR, gamma: float = 1.0, axis  = 1, delta: float = 0.0, eps = 1e-5):
        """Lagrangian of the layer norm activation function"""
        D = x.shape[axis] if axis is not None else x.size
        xmean = x.mean(axis, keepdims=True)
        xmeaned = x - xmean
        y = torch.sqrt(torch.pow(xmeaned, 2).mean(axis, keepdims=True) + eps)
        return (D * gamma * y + (delta * x).sum()).sum()
    
    @staticmethod
    def scale(x: TENSOR, scalar: float):
        l = (scalar*0.5) * torch.pow(x, 2).sum()
        return l