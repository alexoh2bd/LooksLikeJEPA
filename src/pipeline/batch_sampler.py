"""LeJEPA batch sampler for semantically-informed training batches.

Drop-in ``torch.utils.data.Sampler`` that yields batches of size
``batch_size`` where each batch is composed of ``batch_size // K`` complete
clusters, ensuring that every batch contains a dense set of mutual
hard-negative pairs as identified by the B3-style pipeline.

Quick start
-----------
    from src.pipeline.batch_sampler import LeJEPABatchSampler

    sampler = LeJEPABatchSampler(
        clusters_path="outputs/imagenet1k_b3/clusters/clusters.json",
        batch_size=256,
        K=32,
        seed=42,
        epoch=0,
    )
    loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_sampler=sampler,
        num_workers=8,
    )

    # For DDP, increment epoch each round and set rank/world_size:
    sampler = LeJEPABatchSampler(..., rank=dist.get_rank(), world_size=dist.get_world_size())

Integration with Lightning
--------------------------
Override ``train_dataloader`` in your ``LightningModule``::

    def train_dataloader(self):
        sampler = LeJEPABatchSampler(
            clusters_path=self.hparams.clusters_path,
            batch_size=self.hparams.batch_size,
            K=self.hparams.K,
            seed=self.hparams.seed,
            epoch=self.current_epoch,
        )
        return DataLoader(self.train_dataset, batch_sampler=sampler, num_workers=8)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class LeJEPABatchSampler(Sampler[list[int]]):
    """Yield fixed-size batches whose samples come from the same clusters.

    Each batch consists of exactly ``clusters_per_batch = batch_size // K``
    clusters.  Within each selected cluster all ``K`` samples are included
    (possibly sub-sampled if the cluster has been inflated beyond K, but
    Stage 4 guarantees exact-K clusters).

    For multi-GPU (DDP) training each rank receives a disjoint subset of
    the cluster schedule so that no sample is duplicated across ranks.

    Parameters
    ----------
    clusters_path:
        Path to ``clusters.json`` produced by Stage 4.
        Format: ``{"0": [idx0, idx1, ...], "1": [...], ...}``
    batch_size:
        Total number of samples per batch.  Must be divisible by ``K``.
    K:
        Samples per cluster (must match what was used in Stage 4).
    seed:
        Base random seed; epoch number is folded in for per-epoch diversity.
    epoch:
        Current training epoch (used to vary shuffle across epochs).
    drop_last:
        Drop the last incomplete batch if the total sample count is not
        divisible by ``batch_size``.  Defaults to ``True``.
    rank:
        This process's rank in DDP (0 for single-GPU).
    world_size:
        Total number of DDP processes (1 for single-GPU).
    max_batches_total:
        Optional. If set, cap total batches per epoch to this value (must be
        divisible by world_size). Use to match steps/epoch with non-PHN control.
    """

    def __init__(
        self,
        clusters_path: str,
        batch_size: int = 256,
        K: int = 32,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
        max_batches_total: int | None = None,
    ) -> None:
        if batch_size % K != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by K={K}. "
                f"Suggested: batch_size={K * (batch_size // K)} or K={batch_size // (batch_size // K)}"
            )

        self.batch_size = batch_size
        self.K = K
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.clusters_per_batch = batch_size // K
        self.max_batches_total = max_batches_total
        if max_batches_total is not None and max_batches_total % world_size != 0:
            raise ValueError(
                f"max_batches_total={max_batches_total} must be divisible by world_size={world_size}"
            )

        # Load clusters
        with open(clusters_path) as fh:
            raw: dict[str, list[int]] = json.load(fh)

        # Normalise: store as a list of numpy arrays sorted by cluster ID
        self._clusters: list[np.ndarray] = [
            np.array(raw[k], dtype=np.int64)
            for k in sorted(raw.keys(), key=int)
        ]
        self._n_clusters = len(self._clusters)

        logger.info(
            "LeJEPABatchSampler: %d clusters × K=%d  →  %d samples  "
            "(batch_size=%d, clusters_per_batch=%d, rank=%d/%d)%s",
            self._n_clusters, K,
            sum(len(c) for c in self._clusters),
            batch_size, self.clusters_per_batch,
            rank, world_size,
            f"  [capped to {max_batches_total} batches/epoch for control parity]" if max_batches_total is not None else "",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to change the shuffle seed."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of sample indices."""
        rng = random.Random(self.seed + self.epoch * 31337)

        # Shuffle cluster order
        cluster_order = list(range(self._n_clusters))
        rng.shuffle(cluster_order)

        # DDP: each rank takes every world_size-th batch slot
        # Build all batches first, then slice by rank
        all_batches: list[list[int]] = []
        for start in range(0, len(cluster_order) - self.clusters_per_batch + 1, self.clusters_per_batch):
            batch_cluster_ids = cluster_order[start: start + self.clusters_per_batch]
            indices: list[int] = []
            for cid in batch_cluster_ids:
                cluster = self._clusters[cid]
                # Sub-sample exactly K if cluster has been inflated
                if len(cluster) > self.K:
                    chosen = rng.sample(range(len(cluster)), self.K)
                    indices.extend(int(cluster[i]) for i in chosen)
                else:
                    indices.extend(int(x) for x in cluster)
            all_batches.append(indices)

        # Cap total batches if parity with non-PHN is requested
        if self.max_batches_total is not None:
            all_batches = all_batches[: self.max_batches_total]

        # Distribute across DDP ranks
        rank_batches = all_batches[self.rank:: self.world_size]

        if self.drop_last:
            # Ensure each rank has the same number of batches
            n_batches_per_rank = len(all_batches) // self.world_size
            rank_batches = rank_batches[:n_batches_per_rank]

        yield from rank_batches

    def __len__(self) -> int:
        """Number of batches per epoch (for this rank)."""
        total_complete = self._n_clusters // self.clusters_per_batch
        if self.max_batches_total is not None:
            total_complete = min(total_complete, self.max_batches_total)
        batches_per_rank = total_complete // self.world_size
        return batches_per_rank

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def total_samples(self) -> int:
        return sum(len(c) for c in self._clusters)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_clusters={self._n_clusters}, "
            f"K={self.K}, "
            f"batch_size={self.batch_size}, "
            f"epoch={self.epoch}, "
            f"rank={self.rank}/{self.world_size})"
        )


class PosHardNegBatchSampler(LeJEPABatchSampler):
    """Semantics-focused alias for ``LeJEPABatchSampler``.

    The name makes the two-level structure explicit:

    * **Pos** — ``MixedViewDataset`` draws local views from the *positive pool*
      (CLIP neighbor ranks ``[0, p)``), pulling semantically similar images
      into each sample's view set.
    * **HardNeg** — this sampler groups samples from the same METIS cluster
      (built from ranks ``[p, p+m)``) into every batch, ensuring every
      in-batch pair is a mutual hard negative for the contrastive signal.

    No logic changes over ``LeJEPABatchSampler`` — pure rename for clarity
    at call sites that use both components together.

    Parameters
    ----------
    clusters_path:
        Path to ``clusters.json`` produced by Stage 4 of the B3 pipeline.
    batch_size, K, seed, epoch, drop_last, rank, world_size:
        Forwarded unchanged to ``LeJEPABatchSampler``.
    """
class PosBatchSampler(Sampler[list[int]]):
    """Yield fixed-size batches where all samples are drawn uniformly at random.
    
    Unlike LeJEPABatchSampler, this does NOT use cluster structure or hard negatives.
    It simply shuffles the dataset and yields batches, useful for baselines like
    SimCLR or standard contrastive learning without mining.
    
    For multi-GPU (DDP) training, each rank receives a disjoint subset of batches
    so no sample is duplicated across ranks within an epoch.
    
    Parameters
    ----------
    dataset_size:
        Total number of samples in the dataset.
    batch_size:
        Number of samples per batch.
    seed:
        Base random seed; epoch number is folded in for per-epoch diversity.
    epoch:
        Current training epoch (used to vary shuffle across epochs).
    drop_last:
        Drop the last incomplete batch if dataset_size is not divisible
        by batch_size. Defaults to True.
    rank:
        This process's rank in DDP (0 for single-GPU).
    world_size:
        Total number of DDP processes (1 for single-GPU).
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int = 256,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

        logger.info(
            "PosBatchSampler: %d samples, batch_size=%d, rank=%d/%d",
            dataset_size, batch_size, rank, world_size
        )

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to change the shuffle seed."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of sample indices."""
        rng = random.Random(self.seed + self.epoch * 31337)
        
        # Shuffle all indices
        indices = list(range(self.dataset_size))
        rng.shuffle(indices)
        
        # Build all batches
        all_batches: list[list[int]] = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start: start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                all_batches.append(batch)
        
        # Distribute across DDP ranks
        rank_batches = all_batches[self.rank:: self.world_size]
        
        if self.drop_last:
            # Ensure each rank has the same number of batches
            n_batches_per_rank = len(all_batches) // self.world_size
            rank_batches = rank_batches[:n_batches_per_rank]
        
        yield from rank_batches

    def __len__(self) -> int:
        """Number of batches per epoch (for this rank)."""
        if self.drop_last:
            total_complete = self.dataset_size // self.batch_size
        else:
            total_complete = (self.dataset_size + self.batch_size - 1) // self.batch_size
        
        batches_per_rank = total_complete // self.world_size
        return batches_per_rank

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_size={self.dataset_size}, "
            f"batch_size={self.batch_size}, "
            f"epoch={self.epoch}, "
            f"rank={self.rank}/{self.world_size})"
        )