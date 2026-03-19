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


def all_reduce(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    return functional_all_reduce(tensor, op=ReduceOp.SUM) / dist.get_world_size()

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
    
    return losses.mean()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization via the Epps-Pulley test.

    Parameters
    ----------
    M : int
        Number of random projection directions (|A|). Resampled every forward call.
    knots : int
        Quadrature points for the Epps-Pulley integral.
    upper : float
        Upper bound of the (half-)integration domain.  The full domain is
        [-upper, upper]; symmetry of the squared-modulus ECF lets us integrate
        over [0, upper] and double (absorbed into the weight constant).
    """
    def __init__(self, M=1024, knots=17, upper=5.0):
        super().__init__()
        self.M = M
        t = torch.linspace(0, upper, knots, dtype=torch.float32)
        dt = upper / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        
    def forward(self, proj, global_step=0):
        # Generate random projections A on the fly or from buffer
        # Paper specifies resampling at every step for 'sketching'
        g = torch.Generator(device=proj.device)
        g.manual_seed(global_step)
    
        A = torch.randn(proj.size(-1), self.M, generator = g, device=proj.device)
        A = A / A.norm(p=2, dim=0) # Normalize slices to unit sphere
        
        # Project high-dim embeddings to 1D slices
        # (N, D) @ (D, M) -> (N, M) -> (N, M, 1) * (knots) -> (N, M, K)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        # if dist.is_initialized():
        #     cos_mean = all_reduce(cos_mean)
        #     sin_mean = all_reduce(sin_mean)


        # ECF distance calculation
        err = (cos_mean - self.phi).square() + sin_mean.square()
        # Scale by N (batch size) as per the Epps-Pulley statistic definition
        statistic = (err @ self.weights) * proj.size(0) 
        return statistic.mean()

def LeJEPA(all_views_proj, num_global, sigreg_module, lamb=0.05, reg="LeJEPA", target=None, global_step=0):
    """LeJEPA loss: (1 - λ) × Prediction_Loss + λ × SIGReg_Loss.

    Parameters
    ----------
    all_views_proj : Tensor (N, V, D)
        Projections of all views (global + local).
    num_global : int
        Number of leading views that are global (Vg).
    sigreg_module : SIGReg
        Epps-Pulley regulariser module.
    lamb : float
        Weight λ for SIGReg (prediction weight is 1 - λ).
    target : Tensor (N, 1, D), optional
        Pre-computed target embedding (e.g. from an SWA encoder).
        When *None*, the mean of the global views is used.
    global_step : int
        Current global training step. Passed to SIGReg to seed the random
        projection directions identically across all DDP ranks.
    """
    if target is None:
        global_views = all_views_proj[:, :num_global, :]
        target = global_views.mean(dim=1, keepdim=True)
    if reg == "hybrid":
        sim_loss = simclr_loss(all_views_proj[:, :num_global, :], all_views_proj[:, num_global:, :], temperature=0.5)
    else:
        sim_loss = (all_views_proj - target).square().mean()
    reg_loss = sigreg_module(all_views_proj.reshape(-1, all_views_proj.size(-1)), global_step)
    
    total_loss = (1 - lamb) * sim_loss + lamb * reg_loss
    return total_loss, sim_loss, reg_loss


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

    # MSE prediction loss
    inv_loss = (centers - all_proj).square().mean()

    # SIGReg over each view
    # sigreg_losses = []
    # for i in range(all_proj.shape[1]):
    #     view_emb = all_proj[:, i, :]
    #     sigreg_losses.append(sigreg(view_emb, global_step).sum())
    sr_loss = sigreg(all_proj.reshape(-1, all_proj.size(-1)), global_step)
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

