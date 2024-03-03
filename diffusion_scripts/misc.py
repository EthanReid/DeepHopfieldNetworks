import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence
from inspect import isfunction
import math
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms


TENSOR = torch.Tensor


class Lambda(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: TENSOR):
        return self.fn(x)


class Patch(nn.Module):
    def __init__(
        self,
        dim: int = 4,
        *,
        n: int = 28,
    ):
        super().__init__()

        self.transform = Lambda(
            lambda x: rearrange(
                x, "... c (h p1) (w p2) -> ... (h w) (c p1 p2)", p1=dim, p2=dim
            )
        )

        r = n // dim

        self.N = r**2

        self.revert = Lambda(
            lambda x: rearrange(
                x,
                "... (h w) (c p1 p2) -> ... c (h p1) (w p2)",
                h=r,
                w=r,
                p1=dim,
                p2=dim,
            )
        )

    def forward(self, x: TENSOR, *, reverse: bool = False):
        if reverse:
            return self.revert(x)

        return self.transform(x)

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found")
    else:
        print("No MPS or CUDA device found. Falling back to CPU.")
        device = torch.device("cpu")
    return device

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

def avg_loss(avg, loss):
    avg 
