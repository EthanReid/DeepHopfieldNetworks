import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence
import dhn
from dhn.misc import TENSOR


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
        n: int = 32,
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


class EnergyLayerNorm(nn.Module):
    def __init__(self, in_dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(
                1,
            )
        )

        self.bias = nn.Parameter(torch.zeros(in_dim)) if bias else 0.0

    def forward(self, x: TENSOR):
        xu = x.mean(-1, keepdim=True)
        xm = x - xu
        o = xm / torch.sqrt((xm**2.0).mean(-1, keepdim=True) + self.eps)

        return self.gamma * o + self.bias


class PositionEncode(nn.Module):
    def __init__(self, dim: int, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.normal(0.0, 0.002, size=[1, n, dim]))

    def forward(self, x: TENSOR):
        return x + self.weight


class Hopfield(nn.Module):
    def __init__(self, in_dim, multiplier):
        super().__init__()
        self.hop_1 = dhn.layers.Linear(in_dim=in_dim, multiplier=multiplier, lagrangian=dhn.Lagrangian.exp, activation=torch.exp)
        self.hop_2 = dhn.layers.Linear(in_dim=in_dim*multiplier, multiplier=1, lagrangian=dhn.Lagrangian.relu, activation=nn.ReLU())
        #self.hop_3 = dhn.layers.Linear(in_dim=in_dim*multiplier, multiplier=1, lagrangian=dhn.Lagrangian.exp, activation=torch.exp)
    def forward(self, x: TENSOR):
        g, e_g = self.hop_1(x)
        g_2, e_g_2 = self.hop_2(g)
        #_, e_g_3 = self.hop_3(g_2)
        return e_g+e_g_2#+e_g_3

#this implementation just returns the lagragian as energy
class Attention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        beta: Optional[float] = None,
        bias: bool = False,
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0

        self.h, self.d = nheads, qk_dim
        self.beta = beta if beta is not None else 1.0 / (qk_dim**0.5)

        self.wq = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))
        self.wk = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))

        self.bq = nn.Parameter(torch.zeros(qk_dim)) if bias else None
        self.bk = nn.Parameter(torch.zeros(qk_dim)) if bias else None

        self.neuron = dhn.Neuron(lagrangian=dhn.Lagrangian.softmax, activation=nn.Softmax())

    def forward(self, g: TENSOR, mask: Optional[TENSOR] = None):
        q = torch.einsum("...kd, ...hzd -> ...khz", g, self.wq)
        k = torch.einsum("...kd, ...hzd -> ...khz", g, self.wk)

        if self.bq is not None:
            q = q + self.bq
            k = k + self.bk

        # B x H x N x N
        A = torch.einsum("...qhz, ...khz -> ...hqk", q, k)

        if mask is not None:
            A *= mask

        #e = (-1.0 / self.beta) * torch.logsumexp(self.beta * A, dim=-1).sum()
        #g, e = self.neuron(A)
        e = -dhn.Lagrangian.softmax(A)
        return e


class ETBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        hn_mult: float = 4.0,
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0

        self.hn = Hopfield(in_dim, hn_mult)
        self.attn = Attention(in_dim, qk_dim, nheads)

    def energy(
        self,
        g: TENSOR,
        mask: Optional[TENSOR] = None,
    ):
        return self.attn(g, mask) + self.hn(g)

    def forward(
        self,
        g: TENSOR,
        mask: Optional[TENSOR] = None,
    ):
        return self.energy(g, mask)


class ET(nn.Module):
    def __init__(
        self,
        x: TENSOR,
        patch: Union[nn.Module, Callable],
        out_dim: Optional[int] = None,
        tkn_dim: int = 256,
        qk_dim: int = 64,
        nheads: int = 12,
        hn_mult: float = 4.0,
        time_steps: int = 1,
        blocks: int = 1,
    ):
        super().__init__()

        x = patch(x)
        _, n, d = x.shape
        print("init x shape: {}".format(str(x.shape)))

        self.K = time_steps

        self.patch = patch

        self.encode = nn.Sequential(
            nn.Linear(d, tkn_dim),
        )

        self.decode = nn.Sequential(
            nn.LayerNorm(tkn_dim, tkn_dim),
            nn.Linear(tkn_dim, out_dim if out_dim is not None else d),
        )

        self.pos = PositionEncode(tkn_dim, n + 1)

        self.cls = nn.Parameter(torch.ones(1, 1, tkn_dim))

        self.mask = nn.Parameter(torch.normal(0, 0.002, size=[1, 1, tkn_dim]))

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EnergyLayerNorm(tkn_dim),
                        ETBlock(
                            tkn_dim,
                            qk_dim,
                            nheads,
                            hn_mult
                        ),
                    ]
                )
                for _ in range(blocks)
            ]
        )
        
    def visualize(
        self,
        x: TENSOR,
        mask_id: Optional[TENSOR] = None,
        alpha: float = 1.0,
        *,
        attn_mask: Optional[Sequence[TENSOR]] = None,
    ):

        x = self.patch(x)
        x = self.encode(x)

        if mask_id is not None:
            x[:, mask_id] = self.mask
            
        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pos(x)

        energies = []
        embeddings = [self.patch(self.decode(x)[:, 1:], reverse=True)]

        for norm, et in self.blocks:
            for _ in range(self.K):
                g = norm(x)
                dEdg, E = torch.func.grad_and_value(et)(g, attn_mask)

                x = x - alpha * dEdg

                energies.append(E)

                embeddings.append(
                    self.patch(self.decode(x)[:, 1:], reverse=True)
                )
                
        g = norm(x)  
        energies.append(et(g, attn_mask))
        return energies, embeddings
        
    #this works the same as x-dEdg, was missing decode
    def evolve(
        self,
        x: TENSOR,
        alpha: float,
        *,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        return_energy: bool = False,
    ):
        energies = [] if return_energy else None
        _dEdg = None
        for norm, et in self.blocks:
            g = norm(x)
            dEdg, E = torch.func.grad_and_value(et)(g, attn_mask)
            if _dEdg == None:
                _dEdg = dEdg
            else:
                _dEdg += dEdg
                x = x - alpha * dEdg
                x = rearrange(x, "b (h w) c -> b c h w", h=32)#was 28
            if return_energy:
                energies.append(E)
                

        if return_energy:
            g = norm(x)
            E = et(g, attn_mask)
            energies.append(E)
            
        return (alpha*_dEdg), None
        #return x, energies
    
    def forward(
        self,
        x: TENSOR,
        mask_id: Optional[TENSOR] = None,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        *,
        alpha: float = 1.0,
        return_energy: bool = False,
        use_cls: bool = False,
    ):
        
        x = self.patch(x)
        x = self.encode(x)

        if mask_id is not None:
            x[:, mask_id] = self.mask

        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pos(x)
        
        dEdg, _ = self.evolve(
            x, alpha, attn_mask=attn_mask, return_energy=return_energy
        )
        #works now, was missing the decode of dEdg
        x = self.decode(dEdg)
        yh = x[:, :1] if use_cls else x[:, 1:]
        #yh = rearrange(yh, "b (h w) c -> b 1 (h c) w", w=28)
        yh = self.patch(yh,reverse=True)
        #if return_energy:
        #    return yh, energies
        return yh
#it appares that ET is return x-predNoise, which should break becasue diffusion takes x-model_out, which it expects to be prednoise
    #this leads me to believe that predNoise is AllButpredNoise, so taking x-AllButPredNoise give me predNoise. 
    #at the moment, if et return dEdg, the model doesnt learn, but if it returns x-dEdt, it leans...
# SOLUTION: I was not passing dEdg into decode, now it works just like x-dEdg