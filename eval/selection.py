import torch
import cv2
import numpy as np

def laplacian_variance(view):
    """Fastest single metric - measures focus/sharpness"""
    # Compute Laplacian variance
    return cv2.Laplacian(view, cv2.CV_64F).var()

def select_laplacian_views(local_views, n_select=4):
    """
    Select n_select views with high information AND diversity
    """
    scores = [laplacian_variance(v) for v in local_views]
    
    # Get indices of top n_select views
    top_indices = np.argsort(scores)[-n_select:][::-1]
    
    # Optional: filter out very low-information views
    if min_threshold is not None:
        top_indices = [idx for idx in top_indices if scores[idx] >= min_threshold]
        
        # If we filtered out too many, relax threshold
        if len(top_indices) < n_select:
            top_indices = np.argsort(scores)[-n_select:][::-1]
    
    selected_views = [local_views[i] for i in top_indices]
    selected_scores = [scores[i] for i in top_indices]
    
    return [local_views[i] for i in selected_indices]
def estimate_lid(model, local_views, global_views, k=20):
    '''
    local_views: (N, Vl, C, H, W)
    global_views: (N, Vg, C, H, W)
    model: ViT encoder
    k: Number of neighbors to consider
    '''

    with torch.no_grad():
        N, Vl, C, H, W = local_views.shape
        
        # Unbind to list of (N, C, H, W) for Encoder
        local_views_list = local_views.unbind(dim=1)
        global_views_list = global_views.unbind(dim=1)
        
        local_emb, _ = model(local_views_list) # (N, Vl, D)
        global_emb, _ = model(global_views_list) # (N, Vg, D)
    

    distances = [torch.cdist(local_emb[i], global_emb[i]) for i in range(N)]
    distances = torch.stack(distances) # N, Vl, Vg
    return distances, local_emb

def select_diverse_views(distances, local_views, n_select=4):
    """
    distances: (N, Vl, Vg)
    local_views: (N, Vl, C, H, W)
    returns: List of n_select tensors, each (N, C, H, W)
    """
    avg_dist = distances.mean(-1) # (N, Vl)
    selected_idx = avg_dist.topk(k=n_select, dim=1).indices  # (N, n_select)
    
    # Gather selected views
    N, Vl, C, H, W = local_views.shape
    
    # Expand indices for gather: (N, n_select, C, H, W)
    idx_expanded = selected_idx.view(N, n_select, 1, 1, 1).expand(-1, -1, C, H, W)
    
    selected_views = torch.gather(local_views, 1, idx_expanded) # (N, n_select, C, H, W)

    return selected_views

def select_greedy_diverse_views(distances, candidate_emb, local_views, n_select=6):
    """
    Greedy selection: iteratively pick candidate that maximizes
    diversity with respect to BOTH globals AND already-selected locals.
    Vectorized implementation for GPU efficiency.
    """
    N, num_candidates, Vg = distances.shape
    device = distances.device
    C, H, W = local_views.shape[2:]
        
    # Precompute distances between all candidate embeddings (N, num_candidates, num_candidates)
    # This is efficient for small num_candidates (e.g. 20)
    dist_matrix = torch.cdist(candidate_emb, candidate_emb) 
    
    # Distance to globals (N, num_candidates)
    dist_to_globals = distances.mean(dim=-1)
    
    # Mask for selected items (N, num_candidates)
    selected_mask = torch.zeros((N, num_candidates), dtype=torch.bool, device=device)
    
    # To store indices (N, n_select)
    selected_indices = torch.zeros((N, n_select), dtype=torch.long, device=device)
    
    # Running sum of distances to selected items for each candidate (N, num_candidates)
    sum_dist_to_selected = torch.zeros((N, num_candidates), device=device)
    
    for i in range(n_select):
        if i == 0:
            # First step: Maximize distance to globals only
            score = dist_to_globals
        else:
            # Score = Global_Dist + Mean_Local_Dist
            # Mean_Local_Dist = Sum_Dist / i
            score = dist_to_globals + (sum_dist_to_selected / i)
        
        # Mask out already selected to prevent re-selection (set to -inf)
        score = score.masked_fill(selected_mask, -float('inf'))
        
        # Select best candidate for each batch element
        best_idx = score.argmax(dim=1) # (N,)
        selected_indices[:, i] = best_idx
        
        # Update mask
        selected_mask.scatter_(1, best_idx.unsqueeze(1), True)
        
        if i < n_select - 1:
            # Update running sum of distances for the next iteration
            # We add the distance from every candidate to the newly selected candidate (best_idx)
            
            # dist_matrix: (N, num_candidates, num_candidates)
            # best_idx: (N,)
            # gather index: (N, num_candidates, 1) to pick the column corresponding to best_idx
            
            idx_for_gather = best_idx.view(N, 1, 1).expand(N, num_candidates, 1)
            new_dists = dist_matrix.gather(2, idx_for_gather).squeeze(2) # (N, num_candidates)
            
            sum_dist_to_selected += new_dists

    # Gather selected views
    # local_views: (N, num_candidates, C, H, W)
    # selected_indices: (N, n_select)
    
    # Expand indices to match local_views shape: (N, n_select, C, H, W)
    idx_expanded = selected_indices.view(N, n_select, 1, 1, 1).expand(-1, -1, C, H, W)
    selected_views = torch.gather(local_views, 1, idx_expanded)
    
    return selected_views

def select_median_view(distances, local_views, n_select=4):
    avg_dist = distances.mean(-1) # (N, Vl)
    median_dist = avg_dist.median(dim=1, keepdim=True).values # (N, 1)

    N, Vl, C, H, W = local_views.shape

    deviated_dist = torch.abs(avg_dist - median_dist)
    selected_idx = deviated_dist.topk(k=n_select, dim=1, largest=False).indices  # (N, n_select)

    idx_expanded = selected_idx.view(N, n_select, 1, 1, 1).expand(-1, -1, C, H, W)
    
    selected_views = torch.gather(local_views, 1, idx_expanded) # (N, n_select, C, H, W)
    return selected_views
    

# def IoU(A, B):
#     '''
#     finds intersect between two matrices A and B
#     '''
#     H, W = A.shape[-2], A.shape[-1]
#     for 
    

def spatial_diversity_selection(local_views, n_select=4):
    iou_matrix = IoU(local_views, local_views)
    