import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple


def linear_cka(
    X: torch.Tensor, 
    Y: torch.Tensor,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Linear CKA between two representation matrices.
    
    CKA measures similarity between representations by comparing
    their kernel matrices (Gram matrices).
    
    Args:
        X: (n_samples, d_x) - representations from model/layer 1
        Y: (n_samples, d_y) - representations from model/layer 2
        epsilon: numerical stability constant
        
    Returns:
        cka_value: float in [0, 1], where 1 = identical representations
        
    Mathematical Formula:
        CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        
        Where:
        - K = XX^T (gram matrix, shape N×N)
        - L = YY^T (gram matrix, shape N×N)
        - HSIC = Hilbert-Schmidt Independence Criterion
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    
    # Center the representations (subtract mean)
    X_centered = X - X.mean(dim=0, keepdim=True) # (N, d_x)
    Y_centered = Y - Y.mean(dim=0, keepdim=True) # (N, d_y)
    
    # Compute gram (kernel) matrices: K = XX^T, L = YY^T
    # These are N×N matrices (not D×D!)
    K = X_centered @ X_centered.T  # (N, N)
    L = Y_centered @ Y_centered.T  # (N, N)
    
    # Compute HSIC using the matrix formulation
    # HSIC(K, L) = trace(K^T H L H) / (n-1)^2
    # For centered data, this simplifies to: trace(KL) / (n-1)^2
    
    n = X.shape[0]
    stat
    # Numerator: HSIC(K, L)
    hsic_kl = torch.sum(K * L)  # Frobenius inner product = trace(K^T L) -> scalar
    
    # Denominator: sqrt(HSIC(K, K) * HSIC(L, L))
    hsic_kk = torch.sum(K * K)  # trace(K^T K) -> scalar
    hsic_ll = torch.sum(L * L)  # trace(L^T L) -> scalar
    
    # CKA formula
    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + epsilon)
    
    return cka_value.item()

def compute_diagnostics(embeddings, labels=None):
    """
    embeddings: (Batch_Size, Views, Dim) OR (N, Dim)
    labels: Optional labels matching the flattened first dimension
    """
    stats = {}
    
    # 1. EFFECTIVE RANK (The "True" Dimensionality)
    # Handle both 2D (already flattened) and 3D (Batch, Views, Dim) inputs
    if embeddings.ndim == 3:
        z = embeddings.flatten(0, 1).float()
    elif embeddings.ndim == 2:
        z = embeddings.float()
    else:
        raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    # Center the embeddings first
    z = z - z.mean(dim=0)
    
    # SVD (Singular Value Decomposition)
    # S contains the singular values (strengths of each dimension)
    try:
        # svd(z): U (N, N), S (min(N, D),), Vh (D, D)
        _, S, _ = torch.linalg.svd(z, full_matrices=False)
        
        # Normalize singular values to create a probability distribution
        # We square them because Singular Values related to Variance ~ Eigenvalues
        singular_values_sq = S.square()
        p = singular_values_sq / singular_values_sq.sum()
        
        # Shannon Entropy in spectral domain
        # Avoid log(0)
        entropy = -torch.sum(p * torch.log(p + 1e-14))
        
        # Effective Rank = e^Entropy
        effective_rank = torch.exp(entropy).item()
        stats['train_stats/eff_rank'] = effective_rank
        
        # Also log the "Slope" of the spectrum (First vs Last)
        # If this is huge (e.g. 10^6), the space is dominated by 1 dimension
        stats['train_stats/condition_number'] = (S[0] / (S[-1] + 1e-6)).item()
        
        
    except RuntimeError:
        # SVD can sometimes fail on very unstable matrices
        stats['train_stats/eff_rank'] = 0.0

    # 2. FEATURE HEALTH (Variance per dimension)
    # std per dimension (Dim,)
    std_per_dim = z.std(dim=0) # (D,)
    avg_std = std_per_dim.mean().item()
    
    # Count "Dead" dimensions (std < 1e-4)
    dead_dims = (std_per_dim < 1e-4).sum().item()
    stats['train_stats/avg_std'] = avg_std
    stats['train_stats/dead_dims'] = dead_dims

    # 3. CLASS SEPARATION (If labels are provided)
    if labels is not None:
        # Simple separation metric: Distance to Class Center vs Distance to Global Center
        
        # Global Center
        global_center = embeddings.mean(dim=0)
        total_var = (embeddings - global_center).norm(dim=1).mean()
        
        # Intra-Class Variance (Average radius of class clusters)
        # We iterate unique labels (slow, but fine for diagnostics every 100 steps)
        unique_labels = labels.unique()
        intra_vars = []
        for c in unique_labels:
            mask = labels == c
            if mask.sum() > 1:
                cluster = embeddings[mask]
                center = cluster.mean(dim=0)
                intra_vars.append((cluster - center).norm(dim=1).mean())
        
        if len(intra_vars) > 0:
            avg_intra_var = torch.stack(intra_vars).mean()
            # Ratio: Higher is better (Clusters are tight relative to global spread)
            stats['train_stats/separation_ratio'] = (total_var / (avg_intra_var + 1e-6)).item()
    

    return stats



"""
Metrics for analyzing representation geometry in self-supervised learning.
Particularly useful for JEPA-style models.
"""


class RepresentationMetrics:
    """
    Collection of metrics for analyzing learned representation geometry.
    """
    
    @staticmethod
    def local_intrinsic_dimensionality(
        embeddings: torch.Tensor,
        k: int = 20,
        method: str = 'mle'
    ) -> Tuple[float, np.ndarray]:
        """
        Compute Local Intrinsic Dimensionality (LID).
        Expects UNNORMALIZED embeddings (raw features).
        
        Args:
            embeddings: (N, D) tensor.
        """
        embeddings = embeddings.float().cpu().numpy()
        N, D = embeddings.shape
        
        # Adaptive k: ensure k is smaller than N
        k = min(k, N - 1)
        if k < 2:
            return 0.0, np.zeros(N)

        # Compute pairwise distances
        # Note: squareform(pdist) is O(N^2) memory. For N > 20k, this might OOM.
        try:
            dists = squareform(pdist(embeddings, metric='euclidean'))
        except MemoryError:
            print(f"Warning: N={N} too large for full pdist. returning default LID.")
            return 0.0, np.zeros(N)
        
        lid_estimates = []
        
        for i in range(N):
            distances = dists[i]
            # Sort and get k+1 nearest (index 0 is self at dist 0)
            nearest_dists = np.sort(distances)[1:k+1]
            
            if method == 'mle':
                r_k = nearest_dists[-1]
                r_i = nearest_dists[:-1]
                if r_k < 1e-10:
                    lid = D 
                else:
                    # Regular MLE: lid = k / sum(log(r_k / r_i))
                    # r_k / r_i: (k-1,) vector of ratios > 1.0
                    # log: (k-1,) vector of positive log ratios
                    lid = 1.0 / np.mean(np.log(r_k / (r_i + 1e-10)))
            elif method == 'mom':
                mean_dist = np.mean(nearest_dists)
                var_dist = np.var(nearest_dists)
                if var_dist < 1e-10:
                    lid = 0.0
                else:
                    lid = mean_dist**2 / (2 * var_dist)
            
            lid_estimates.append(lid)
        
        lid_per_point = np.array(lid_estimates)
        mean_lid = np.mean(lid_per_point)
        
        return mean_lid, lid_per_point
    
    @staticmethod
    def alignment_uniformity(
        embeddings: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
        t: float = 2.0
    ) -> Tuple[float, float]:
        """
        Compute Alignment and Uniformity metrics (Wang & Isola, 2020).
        
        Alignment: How close positive pairs are (lower is better)
        Uniformity: How uniformly distributed embeddings are on hypersphere (lower is better)
        
        Args:
            embeddings: (N, D) tensor of L2-normalized representations
            positive_pairs: (M, 2) tensor of indices for positive pairs
                           If None, assumes consecutive pairs (i, i+1) for even N
            t: temperature parameter for uniformity (default: 2.0)
            
        Returns:
            alignment: Average distance between positive pairs
            uniformity: Measure of distribution uniformity
            
        Reference:
            "Understanding Contrastive Representation Learning through Alignment and Uniformity"
            Wang & Isola, ICML 2020
        """
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        N = embeddings.shape[0]
        
        # === ALIGNMENT ===
        if positive_pairs is None:
            # Default: assume consecutive pairs (0,1), (2,3), ...
            if N % 2 != 0:
                raise ValueError("If positive_pairs is None, N must be even")
            positive_pairs = torch.stack([
                torch.arange(0, N, 2),
                torch.arange(1, N, 2)
            ], dim=1)
        
        # Get embeddings for positive pairs
        z_i = embeddings[positive_pairs[:, 0]] # (M, D)
        z_j = embeddings[positive_pairs[:, 1]] # (M, D)
        
        # Alignment: E[(z_i - z_j)^2]
        # (M, D) - (M, D) -> (M, D) -> norm -> (M,) -> mean -> scalar
        alignment = (z_i - z_j).norm(p=2, dim=1).pow(2).mean()
        
        # === UNIFORMITY ===
        # Compute pairwise dot products efficiently
        # uniformity = log E[e^(-t||z_i - z_j||^2)]
        
        # For efficiency with large N, sample pairs
        if N > 1000:
            # Sample 1000 random pairs
            idx = torch.randint(0, N, (1000, 2)) # (1000, 2)
            z_sample = embeddings[idx] # (1000, 2, D)
            # z_sample[:, 0]: (1000, D), z_sample[:, 1]: (1000, D)
            # (1000,) squared distances
            pairwise_sq_dist = (z_sample[:, 0] - z_sample[:, 1]).norm(p=2, dim=1).pow(2)
        else:
            # Compute all pairs: ||z_i - z_j||^2 = 2 - 2*<z_i, z_j> (since normalized)
            dot_products = embeddings @ embeddings.T # (N, N)
            pairwise_sq_dist = 2 - 2 * dot_products # (N, N)
            # Get upper triangular (excluding diagonal)
            mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
            pairwise_sq_dist = pairwise_sq_dist[mask] # (N*(N-1)/2,)
        
        uniformity = torch.log(torch.exp(-t * pairwise_sq_dist).mean())
        
        return alignment.item(), uniformity.item()
    
    @staticmethod
    def cluster_quality_metrics(
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute Silhouette Score and Davies-Bouldin Index.
        
        These measure cluster quality:
        - Silhouette: [-1, 1], higher is better (1 = perfect clusters)
        - Davies-Bouldin: [0, ∞), lower is better (0 = perfect separation)
        
        Args:
            embeddings: (N, D) tensor of representations
            labels: (N,) tensor of cluster/class labels
            
        Returns:
            dict with 'silhouette' and 'davies_bouldin' scores
        """
        embeddings = embeddings.float().cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Check we have enough samples per class
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 different labels")
        
        # Silhouette Score
        silhouette = silhouette_score(embeddings, labels, metric='euclidean') # scalar
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(embeddings, labels) # scalar
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    @staticmethod
    def uniformity(
        embeddings: torch.Tensor,
        t: float = 2.0,
        max_samples: int = 2000
    ) -> float:
        """
        Compute Uniformity (Wang & Isola, 2020) efficiently.
        Can be run on Test data (does not require positive pairs).
        
        Args:
            embeddings: (N, D) tensor.
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        N = embeddings.shape[0]
        
        # If N is large, sample pairs to avoid O(N^2) memory OOM
        if N > max_samples:
            # Randomly sample indices for pair calculation
            idx1 = torch.randint(0, N, (max_samples, ), device=embeddings.device) # (K,)
            idx2 = torch.randint(0, N, (max_samples, ), device=embeddings.device) # (K,)
            
            z1 = embeddings[idx1] # (K, D)
            z2 = embeddings[idx2] # (K, D)
            
            # Squared Euclidean distance on normalized sphere: ||x-y||^2
            pairwise_sq_dist = (z1 - z2).norm(p=2, dim=1).pow(2) # (K,)
        else:
            # Full pairwise matrix
            dot_products = embeddings @ embeddings.T # (N, N)
            pairwise_sq_dist = 2 - 2 * dot_products # (N, N)
            # Exclude diagonal (distance to self is 0)
            mask = torch.eye(N, device=embeddings.device).bool()
            pairwise_sq_dist = pairwise_sq_dist[~mask]
        
        uniformity = torch.log(torch.exp(-t * pairwise_sq_dist).mean())
        return uniformity.item()
    @staticmethod
    def effective_rank(
        embeddings: torch.Tensor,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute Effective Rank using squared singular values (eigenvalues of covariance).
        
        Measures how many dimensions are effectively used.
        Lower rank = more dimensional collapse.
        
        Args:
            embeddings: (N, D) tensor.
            epsilon: numerical stability constant
            
        Returns:
            effective_rank: Value between 1 and min(N, D)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        # Center the embeddings
        embeddings = embeddings.float()
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute SVD
        # We want eigenvalues of covariance matrix (S^2), so we square singular values
        # svd(embeddings): U (N, N), S (min(N, D),), Vh (D, D)
        _, S, _ = torch.linalg.svd(embeddings, full_matrices=False)
        S = S.square() # (min(N, D),) - Eigenvalues of covariance
        
        # Normalize singular values to get probability distribution
        S_normalized = S / (S.sum() + epsilon) # (min(N, D),)
        
        # Compute entropy
        # entropy: scalar
        entropy = -(S_normalized * torch.log(S_normalized + epsilon)).sum()
        
        # Effective rank = exp(entropy)
        eff_rank = torch.exp(entropy)
        
        return eff_rank.item()
    
    @staticmethod
    def fisher_ratio(
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute Fisher Ratio (between-class variance / within-class variance).
        
        Higher is better - indicates well-separated, compact clusters.
        
        Args:
            embeddings: (N, D) tensor.
            labels: (N,) tensor of class labels
            
        Returns:
            fisher_ratio: Higher means better separation
        """
        embeddings = embeddings.cpu()
        labels = labels.cpu()
        
        unique_labels = torch.unique(labels)
        C = len(unique_labels)  # Number of classes
        D = embeddings.shape[1]  # Dimensionality
        
        # Global mean
        global_mean = embeddings.mean(dim=0) # (D,)
        
        # Between-class scatter
        S_B = torch.zeros(D, D)
        # Within-class scatter
        S_W = torch.zeros(D, D)
        
        for label in unique_labels:
            mask = labels == label
            class_samples = embeddings[mask] # (n_k, D)
            n_k = class_samples.shape[0]
            
            # Class mean
            class_mean = class_samples.mean(dim=0) # (D,)
            
            # Between-class scatter
            mean_diff = (class_mean - global_mean).unsqueeze(1) # (D, 1)
            S_B += n_k * (mean_diff @ mean_diff.T) # scalar * (D, 1) @ (1, D) -> (D, D)
            
            # Within-class scatter
            centered = class_samples - class_mean.unsqueeze(0) # (n_k, D) - (1, D) -> (n_k, D)
            S_W += centered.T @ centered # (D, n_k) @ (n_k, D) -> (D, D)
        
        # Fisher ratio = trace(S_B) / trace(S_W)
        trace_B = torch.trace(S_B) # scalar
        trace_W = torch.trace(S_W) # scalar
        
        fisher = (trace_B / (trace_W + 1e-10)).item()
        
        return fisher

def example_usage():
    """
    Demonstrate how to use these metrics with your LeJEPA training.
    """
    print("Example: Analyzing 2-view vs 12-view representations\n")
    
    # Simulate embeddings from two different training regimes
    torch.manual_seed(42)
    N, D = 1000, 384
    
    # Model trained with 2 views (looser clusters)
    embeddings_2view_unnorm = torch.randn(N, D)
    embeddings_2view = F.normalize(embeddings_2view_unnorm, p=2, dim=1)
    
    # Model trained with 12 views (tighter clusters)
    # Simulate tighter structure by adding class structure
    n_classes = 100
    labels = torch.repeat_interleave(torch.arange(n_classes), N // n_classes)
    
    embeddings_12view = []
    for c in range(n_classes):
        # Generate tight cluster
        class_center = torch.randn(D) * 5
        class_samples = class_center + torch.randn(N // n_classes, D) * 0.3
        embeddings_12view.append(class_samples)
    embeddings_12view_unnorm = torch.cat(embeddings_12view)
    embeddings_12view = F.normalize(embeddings_12view_unnorm, p=2, dim=1)
    
    metrics = RepresentationMetrics()
    
    # === Local Intrinsic Dimensionality ===
    print("=== Local Intrinsic Dimensionality (LID) ===")
    lid_2view, _ = metrics.local_intrinsic_dimensionality(embeddings_2view_unnorm, k=20)
    lid_12view, _ = metrics.local_intrinsic_dimensionality(embeddings_12view_unnorm, k=20)
    print(lid_2view.shape)
    print(lid_12view.shape)
    print(f"2-view LID:  {lid_2view:.2f}")
    print(f"12-view LID: {lid_12view:.2f}")
    print(f"Lower LID → tighter local structure ✓\n")
    
    # === Alignment & Uniformity ===
    print("=== Alignment & Uniformity ===")
    # Create positive pairs (simulate augmented views)
    positive_pairs = torch.stack([
        torch.arange(0, N, 2),
        torch.arange(1, N, 2)
    ], dim=1)
    
    align_2view, unif_2view = metrics.alignment_uniformity(embeddings_2view, positive_pairs)
    align_12view, unif_12view = metrics.alignment_uniformity(embeddings_12view, positive_pairs)
    
    print(f"2-view:  Alignment={align_2view:.4f}, Uniformity={unif_2view:.4f}")
    print(f"12-view: Alignment={align_12view:.4f}, Uniformity={unif_12view:.4f}")
    print(f"Lower alignment → tighter positive pairs ✓\n")
    
    # === Cluster Quality ===
    print("=== Cluster Quality (with pseudo-labels) ===")
    cluster_2view = metrics.cluster_quality_metrics(embeddings_2view, labels)
    cluster_12view = metrics.cluster_quality_metrics(embeddings_12view, labels)
    
    print(f"2-view:  Silhouette={cluster_2view['silhouette']:.4f}, DB={cluster_2view['davies_bouldin']:.4f}")
    print(f"12-view: Silhouette={cluster_12view['silhouette']:.4f}, DB={cluster_12view['davies_bouldin']:.4f}")
    print(f"Higher silhouette + Lower DB → better clusters ✓\n")
    
    # === Effective Rank ===
    print("=== Effective Rank ===")
    erank_2view = metrics.effective_rank(embeddings_2view)
    erank_12view = metrics.effective_rank(embeddings_12view)
    print(f"2-view:  {erank_2view:.2f}")
    print(f"12-view: {erank_12view:.2f}")
    print(f"(Similar values suggest no dimensional collapse)\n")
    
    # === Fisher Ratio ===
    print("=== Fisher Ratio (Inter/Intra-class variance) ===")
    fisher_2view = metrics.fisher_ratio(embeddings_2view, labels)
    fisher_12view = metrics.fisher_ratio(embeddings_12view, labels)
    print(f"2-view:  {fisher_2view:.2f}")
    print(f"12-view: {fisher_12view:.2f}")
    print(f"Higher ratio → better class separation ✓\n")


if __name__ == "__main__":
    example_usage()