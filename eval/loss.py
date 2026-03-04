'''
Loss Functions 

SIGReg (LeJEPA)
VICReg
INFO_NCE (SimCLR)
'''
import torch
# from .base import UnivariateTest
from torch import distributed as dist


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


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, op)
    else:
        return x
def simclr_loss(global_proj, local_proj, labels=None, temperature=0.5):
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
    losses = -pos_sim + denom.unsqueeze(1).log()  # [N, V_l]
    
    return losses.mean()
'''
def simclr_loss(global_proj, local_proj, labels, temperature=0.5, cheating=False):
    
    Actually accidental (SupCon) Supervised Contrastive loss with
    Label-based positive view selection, (rather than all positives).
    
    if is_dist_avail_and_initialized():
        pass
    N, V_g, D = global_proj.shape
    V_l = local_proj.shape[1]
    device = global_proj.device
            
    # Anchors and locals
    anchors = global_proj.mean(dim=1)  # [N, D]
    all_local = local_proj.reshape(N, V_l, D)  # [N, V_l, D]
    
    # Compute all similarities: [N, N, V_l]
    sim = F.cosine_similarity(
        anchors.unsqueeze(1).unsqueeze(2),      # [N, 1, 1, D]
        all_local.unsqueeze(0),                  # [1, N, V_l, D]
        dim=3
    ) / temperature

    # Vectorized loss computation
    exp_sim = torch.exp(sim)  # [N, N, V_l]
    sum_over_vl = exp_sim.sum(dim=2)  # [N, N]

    # if cheating:
    #     labels_expanded = labels.unsqueeze(1)  # [N, 1]
    #     same_class = (labels_expanded == labels_expanded.T)  # [N, N]
    #     same_index = torch.eye(N, dtype=torch.bool, device=device)  # [N, N]
    #     neg_mask = ~same_class & ~same_index  # [N, N]
    #     neg_sum = (sum_over_vl * neg_mask.float()).sum(dim=1)  # [N]
    # else:
    neg_sum = sum_over_vl.sum(dim=1)
    
    indices = torch.arange(N, device=device)
    pos_sim = sim[indices, indices, :]  # [N, V_l]
    pos_exp = exp_sim[indices, indices, :]  # [N, V_l]
    pos_sum = pos_exp.sum(dim=1)  # [N]

    denom = pos_sum + neg_sum  # [N]

    # Loss per sample: [N]
    losses = (denom.log().unsqueeze(1) - pos_sim).mean(dim=1)

    # Mask samples with no negatives
    mask = neg_sum > 0
    if mask.any():
        return losses[mask].mean()
    else:
        return torch.tensor(0.0, device=device)
    
    # # Compute loss per sample
    # losses = []
    # for i in range(N):
    #     # Positives: own views
    #     pos_sim = sim[i, i, :]  # [V_l]
        
    #     # Negatives: views from different class samples
    #     neg_samples = neg_mask[i]  # [N] boolean mask
    #     neg_sim = sim[i, neg_samples, :].reshape(-1)  # [num_negs * V_l]
        
    #     if neg_sim.numel() == 0:
    #         # No negatives for this sample (entire batch is same class)
    #         continue
        
    #     # InfoNCE for each positive view
    #     pos_exp = torch.exp(pos_sim)  # [V_l]
    #     neg_exp_sum = torch.exp(neg_sim).sum()  # scalar
        
    #     # Loss for all positives of sample i
    #     loss_i = -torch.log(pos_exp / (pos_exp + neg_exp_sum))  # [V_l]
    #     losses.append(loss_i.mean())
    # Pre-compute masks
'''
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()



def LeJEPA(global_proj, all_proj, sigreg, lamb, losstype="LeJEPA", labels=None, global_step=None):
    """
    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    sigreg: Sigreg module
    lamb: scalar weight    
    """
    # Centers from global views
    centers = global_proj.mean(dim=1, keepdim=True) # (N, 1, D)
    
    # Prediction loss (MSE between centers and all views)
    # if losstype == "hybrid":
    #     local_proj = all_proj[:, global_proj.shape[1]:, :] # (N, Vl, D)
    #     sim_loss= simclr_loss(global_proj,local_proj, labels, temperature=0.5)
    # else:
    sim_loss = (centers - all_proj).square().mean()
    
    sigreg_losses = []
    for i in range(all_proj.shape[1]):
        view_emb = all_proj[:, i, :] # (N, D)
        l = sigreg(view_emb) # scalar
        l = l.sum()
        sigreg_losses.append(l)
    sigreg_loss = torch.stack(sigreg_losses).mean()
    
    return (1 - lamb) * sim_loss + lamb * sigreg_loss, sim_loss, sigreg_loss





class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, M=256):
        # proj: (N*V, D) - flattened batch of projections
        # with torch.no_grad():
            # step = dist.all_reduce(local_step,o)
        A = torch.randn(proj.size(-1), M, device=proj.device, dtype=proj.dtype) # (D, 256)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t # (N*V, 256, 1) * (knots,) -> (N*V, 256, knots)
        # err: Mean over batch (dim 0) -> (256, knots)
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        # Global reduction for DDP (average across processes)
        if is_dist_avail_and_initialized():
            all_reduce(cos_mean)  # Now averages globally
            all_reduce(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * proj.size(0) # (256,) * scalar -> scalar (after mean)
        return statistic.mean()


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

