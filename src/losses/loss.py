'''
Loss Functions 

SIGReg (LeJEPA)
VICReg
INFO_NCE (SimCLR)
'''
import torch
# from .base import UnivariateTest
from torch import distributed as dist
from typing import Optional, Tuple


import torch
import timm 
from torchvision.transforms import v2
from torchvision.ops import MLP
from datasets import load_dataset
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np


from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp

from losses.lejepa import SlicedEppsPulley


def all_reduce(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    return functional_all_reduce(tensor, op=ReduceOp.SUM) / dist.get_world_size()

    
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def _distributed_elementwise_mean(tensor):
    if not is_dist_avail_and_initialized():
        return tensor.mean()
    # functional_all_reduce is autograd-aware (backward distributes sum-grad to all ranks)
    # divide by global_count for the mean, but do NOT also divide by world_size since
    # DDP gradient averaging will do that automatically
    N = tensor.numel()
    s = functional_all_reduce(tensor.sum().unsqueeze(0), op=ReduceOp.SUM)
    return s[0] / (N * dist.get_world_size())



def simclr_loss(global_proj, local_proj, temperature=0.5):
    '''
    SimCLR/InfoNCE loss: global views as anchors, local views as positives
    '''
    N, V_g, D = global_proj.shape
    V_l = local_proj.shape[1]
    device = global_proj.device
    
    # Anchors: mean of global views [N, D]
    anchors = global_proj.mean(dim=1)
    
    # All local views [N, V_l, D]
    all_local = local_proj
    
    # Compute similarities: [N, N, V_l]
    sim = F.cosine_similarity(
        anchors.unsqueeze(1).unsqueeze(2),  # [N, 1, 1, D]
        all_local.unsqueeze(0),              # [1, N, V_l, D]
        dim=3
    ) / temperature
    
    # Create mask for positives (diagonal samples)
    eye_mask = torch.eye(N, dtype=torch.bool, device=device)  # [N, N]
    
    # Extract positive similarities: [N, V_l]
    pos_sim = sim[eye_mask]  # Shape: [N, V_l]
    
    # For denominator, we need exp of all similarities
    exp_sim = torch.exp(sim)  # [N, N, V_l]
    
    # Sum over all samples and views for each anchor
    denom = exp_sim.sum(dim=(1, 2))  # [N]
    
    # Loss: -log(exp(pos) / sum(exp(all)))
    # = -pos + log(sum(exp(all)))
    # Average over all V_l positive views

    # weight denominator differently
    # pos sim + sigreg + weight denom loss + probe
    losses = -pos_sim + denom.unsqueeze(1).log()  # [N, V_l]

    return _distributed_elementwise_mean(losses)


class SIGReg(nn.Module):
    def __init__(self, M=1024, knots=17, upper=5.0):  # upper=5.0, integrate [0,5] x2 via symmetry
        super().__init__()
        self.M = M
        t = torch.linspace(0, upper, knots)
        dt = upper / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        # x2 for symmetry: ∫_{-5}^{5} = 2 * ∫_{0}^{5} since integrand is even in t
        self.register_buffer("weights", weights * window * 2.0)

    def forward(self, proj, global_step=0):
        # proj: (N_local, D) — local batch on this GPU
        g = torch.Generator(device=proj.device)
        g.manual_seed(global_step)   # same seed = same A across all ranks
        A = torch.randn(proj.size(-1), self.M, generator=g, device=proj.device)
        A = A / A.norm(p=2, dim=0)

        x_t = (proj @ A).unsqueeze(-1) * self.t   # (N, M, knots)

        # Compute local ECF means (supports autograd gradient flow)
        cos_mean = x_t.cos().mean(0)   # (M, knots) — local ECF real part
        sin_mean = x_t.sin().mean(0)   # (M, knots) — local ECF imaginary part

        # Average across DDP ranks via functional all_reduce (autograd-compatible)
        # Unlike dist.all_reduce (in-place, breaks gradient graph), this preserves grad_fn
        cos_mean = all_reduce(cos_mean)  # global mean real part of ECF
        sin_mean = all_reduce(sin_mean)  # global mean imaginary part of ECF

        N = float(proj.size(0))
        if dist.is_initialized():
            N = N * dist.get_world_size()

        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * N   # scale by global N per Epps-Pulley statistic
        return statistic.mean()


def LeJEPA(all_views_proj, num_global, sigreg_module, lamb=0.05,
           reg="LeJEPA", target=None, global_step=0):
    # all_views_proj: (N, V, D)
    N, V, D = all_views_proj.shape

    # Prediction loss
    if target is None:
        target = all_views_proj[:, :num_global, :].mean(dim=1, keepdim=True)  # (N, 1, D)

    if reg == "hybrid":
        sim_loss = simclr_loss(all_views_proj[:, :num_global, :],
                               all_views_proj[:, num_global:, :], temperature=0.5)
    else:
        sim_loss = _distributed_elementwise_mean((all_views_proj - target).square())

    # SIGReg: once per view, over N samples — matches paper Algorithm 2
    # all_views_proj: (N, V, D) -> iterate over V dimension
    sigreg_per_view = torch.stack([
        sigreg_module(all_views_proj[:, v, :], global_step)
        for v in range(V)
    ])  # (V,)
    reg_loss = sigreg_per_view.mean()

    total_loss = (1 - lamb) * sim_loss + lamb * reg_loss
    return total_loss, sim_loss, reg_loss


def compute_author_lejepa_loss(
    all_proj: torch.Tensor,
    n_global: int,
    sliced_ep: nn.Module,
    lamb: float,
    combine: str = "additive",
    target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match author ``LeJEPA._compute_loss`` in ``losses/lejepa.py`` with layout ``(N, V, D)``.

    Author reference (layout ``(V, N, K)``)::

        sigreg_loss = sigreg(all_projected.reshape(-1, all_projected.size(-1)))

    ``SlicedEppsPulley`` is called with a single tensor argument only; random ``A`` and the
    internal step buffer behave like the author's ``LeJEPA`` module (do not pass Lightning
    ``global_step`` into ``SlicedEppsPulley``).

    **DDP:** Invariance uses :func:`_distributed_elementwise_mean` (global mean over ranks).
    SIGReg/Epps-Pulley paths all-reduce slice statistics. Assumes synchronous training steps
    across ranks and symmetric per-rank batch sizes (prefer ``drop_last=True`` on train).
    If ranks ever desync on step count, sliced projections could diverge; see
    ``SlicedEppsPulley`` docstring.
    """
    if target is None:
        center = all_proj[:, :n_global, :].mean(dim=1, keepdim=True)
    else:
        center = target

    inv_loss = _distributed_elementwise_mean((all_proj - center).square())

    flat = all_proj.reshape(-1, all_proj.size(-1))
    sigreg_loss = sliced_ep(flat)

    if combine == "additive":
        total = inv_loss + lamb * sigreg_loss
    elif combine == "convex":
        total = (1.0 - lamb) * inv_loss + lamb * sigreg_loss
    else:
        raise ValueError(f"combine must be 'additive' or 'convex', got {combine!r}")

    return total, inv_loss, sigreg_loss


# weighted hybrid between InfoNCE and MSE + SIGReg
def weighted_hybrid(global_proj, all_proj, sigreg, w=0.5, lamb=0.05, global_step=0):
    """
    True hybrid: w * (MSE + SIGReg) + (1 - w) * InfoNCE

    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    sigreg: SIGReg module
    w: weight for LeJEPA component (1-w goes to InfoNCE)
    lamb: SIGReg weight within the LeJEPA component
    """
    centers = global_proj.mean(dim=1, keepdim=True)  # (N, 1, D)

    # InfoNCE between global anchors and local views
    local_proj = all_proj[:, global_proj.shape[1]:, :]  # (N, Vl, D)
    cl_loss = simclr_loss(global_proj, local_proj, temperature=0.5)

    # MSE prediction loss (global mean across ranks)
    inv_loss = _distributed_elementwise_mean((centers - all_proj).square())

    # SIGReg over each view
    # sigreg_losses = []
    # for i in range(all_proj.shape[1]):
    #     view_emb = all_proj[:, i, :]
    #     sigreg_losses.append(sigreg(view_emb, global_step).sum())
    flat = all_proj.reshape(-1, all_proj.size(-1))
    if isinstance(sigreg, SlicedEppsPulley):
        sr_loss = sigreg(flat)
    else:
        sr_loss = sigreg(flat, global_step)
    lejepa_loss = (1 - lamb) * inv_loss + lamb * sr_loss
    total_loss = w * lejepa_loss + (1 - w) * cl_loss

    return total_loss, lejepa_loss, cl_loss, sr_loss




def VICReg(global_proj, all_proj, lamb=25,mu=25,nu=1, gamma=1.0, eps = 0.0001):
    """
    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    lamb: scalar weight
    """ 
    def variance_loss(Z):
        var = Z.var(dim=0) # (D,)
        std_z = torch.sqrt(var+ eps)

        varloss = torch.mean(torch.relu (gamma - std_z))
        return varloss
    def cov_loss(Z):
        '''
        Z: N, D
        '''
        N, D = Z.shape
        Z_centered = Z - Z.mean(dim=0,keepdim=True) # (N, D)
        cov = (Z_centered.T @ Z_centered)/ (N-1) #D, D
        # Off-diagonal elements only
        off_diag_mask = ~torch.eye(D, dtype=bool, device=Z.device)
        cov_loss = (cov [off_diag_mask] ** 2).sum() / D
        
        return cov_loss
    def invariance_loss(Z, Z_prime):
        return (Z - Z_prime).square().mean()


    N, V, D = all_proj.shape
    Vg = global_proj.shape[1]
    Vl = V-Vg
    # Centers from global views
    centers = global_proj.mean(dim=1) # (N, D)
    # Prediction loss (MSE between centers and all views)
    vicreg_losses = []
    # Compare Local Views to Global Center Views
    for i in range(V):
        view_emb = all_proj[ :, i,  :] # (N, D)
        
        variance = variance_loss(view_emb) + variance_loss(centers)
        covariance = cov_loss(view_emb) + cov_loss(centers)

        invariance = invariance_loss(view_emb, centers)

        l = (lamb * invariance) + (mu * variance) + (nu * covariance)
        vicreg_losses.append(l)
    return torch.stack(vicreg_losses).mean()

