'''
Loss Functions 

SIGReg (LeJEPA)
VICReg
INFO_NCE (SimCLR)
'''

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

'''
def simclr_loss(global_proj, local_proj, labels, temperature=0.5):
    """Standard InfoNCE loss for SimCLR"""
    N = global_proj.shape[0]
    device = global_proj.device
    
    # 1. Create anchors (mean of global views per sample)
    anchors = global_proj.mean(dim=1)  # [N, D]
    
    # 2. Flatten all local views into single matrix
    all_local = local_proj.reshape(N, -1, local_proj.shape[-1])  # [N, V_total, D]
    V_total = all_local.shape[1]
    
    total_loss = 0.0
    
    for i in range(N):
        anchor = anchors[i]  # [D]
        
        # 3. Positives: all local views from sample i
        positives = all_local[i]  # [V_total, D]
        
        # 4. Negatives: all local views from OTHER samples
        # Create mask for all samples except i
        neg_mask = torch.ones(N, dtype=torch.bool, device=device)
        neg_mask[i] = False
        negatives = all_local[neg_mask].reshape(-1, all_local.shape[-1])  # [(N-1)*V_total, D]
        
        # 5. Compute similarities (cosine)
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positives, dim=1)  # [V_total]
        neg_sim = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=1)  # [(N-1)*V_total]
        
        # 6. InfoNCE loss
        pos_exp = torch.exp(pos_sim / temperature)  # [V_total]
        neg_exp = torch.exp(neg_sim / temperature)  # [(N-1)*V_total]
        
        # For each positive, compute loss against ALL negatives
        for p_exp in pos_exp:
            loss_i = -torch.log(p_exp / (p_exp + neg_exp.sum()))
            total_loss += loss_i
    
    # Average over all positive pairs
    return total_loss / (N * V_total)
'''
'''
def simclr_loss(global_proj, local_proj, temperature=0.5):
    """
    Fully vectorized SimCLR loss - no loops!
    """
    N, V_g, D = global_proj.shape
    V_l = local_proj.shape[1]
    
    # Anchors: [N, D]
    anchors = global_proj.mean(dim=1)
    
    # Flatten locals: [N*V_l, D]
    locals_flat = local_proj.reshape(-1, D)
    
    # Compute all pairwise similarities: [N, N*V_l]
    sim_matrix = F.cosine_similarity(
        anchors.unsqueeze(1),           # [N, 1, D]
        locals_flat.unsqueeze(0),       # [1, N*V_l, D]
        dim=2
    ) / temperature
    
    # Create positive mask: [N, N*V_l]
    # Positives are at indices [i*V_l : (i+1)*V_l] for anchor i
    pos_mask = torch.zeros(N, N * V_l, dtype=torch.bool, device=anchors.device)
    for i in range(N):
        pos_mask[i, i*V_l:(i+1)*V_l] = True
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    
    # For each anchor, sum positives and all (pos + neg)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)  # [N]
    all_sum = exp_sim.sum(dim=1)                # [N]
    
    loss = -torch.log(pos_sum / all_sum).mean()
    return loss
'''
def simclr_loss(global_proj, local_proj, labels, temperature=0.5):
    N, V_g, D = global_proj.shape
    V_l = local_proj.shape[1]
    device = global_proj.device
    
    # Pre-compute masks
    labels_expanded = labels.unsqueeze(1)  # [N, 1]
    same_class = (labels_expanded == labels_expanded.T)  # [N, N]
    same_index = torch.eye(N, dtype=torch.bool, device=device)  # [N, N]
    neg_mask = ~same_class & ~same_index  # [N, N]
    
    # Anchors and locals
    anchors = global_proj.mean(dim=1)  # [N, D]
    all_local = local_proj.reshape(N, V_l, D)  # [N, V_l, D]
    
    # Compute all similarities: [N, N, V_l]
    sim = F.cosine_similarity(
        anchors.unsqueeze(1).unsqueeze(2),      # [N, 1, 1, D]
        all_local.unsqueeze(0),                  # [1, N, V_l, D]
        dim=3
    ) / temperature
    
    # Compute loss per sample
    losses = []
    for i in range(N):
        # Positives: own views
        pos_sim = sim[i, i, :]  # [V_l]
        
        # Negatives: views from different class samples
        neg_samples = neg_mask[i]  # [N] boolean mask
        neg_sim = sim[i, neg_samples, :].reshape(-1)  # [num_negs * V_l]
        
        if neg_sim.numel() == 0:
            # No negatives for this sample (entire batch is same class)
            continue
        
        # InfoNCE for each positive view
        pos_exp = torch.exp(pos_sim)  # [V_l]
        neg_exp_sum = torch.exp(neg_sim).sum()  # scalar
        
        # Loss for all positives of sample i
        loss_i = -torch.log(pos_exp / (pos_exp + neg_exp_sum))  # [V_l]
        losses.append(loss_i.mean())
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

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
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.size(0) # (256,) * scalar -> scalar (after mean)
        return statistic.mean()

def LeJEPA(global_proj, all_proj, sigreg, lamb, global_step=None):
    """
    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    lamb: scalar weight
    """
    # Centers from global views
    centers = global_proj.mean(dim=1, keepdim=True) # (N, 1, D)
    
    # Prediction loss (MSE between centers and all views)
    # (N, 1, D) - (N, V, D) -> (N, V, D) -> scalar mean
    sim_loss = (centers - all_proj).square().mean()
    
    # Vectorized SIGReg: flatten views dimension and process all at once
    # This avoids the Python loop which keeps intermediate tensors alive
    N, V, D = all_proj.shape
    sigreg_losses = []
    for i in range(V):
        view_emb = all_proj[:, i, :] # (N, D)
        l = sigreg(view_emb) # scalar
        sigreg_losses.append(l)
    sigreg_loss = torch.stack(sigreg_losses).mean()
    
    return (1 - lamb) * sim_loss + lamb * sigreg_loss, sim_loss, sigreg_loss


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

