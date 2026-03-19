"""Thread-safe memory-mapped index over precomputed CLIP/teacher neighbor rankings.

Wraps the two numpy arrays produced by Stage 1 + Stage 2 of the B3 pipeline:
  - ``neighbor_indices.npy``  shape ``(N, top_k)`` int32 — neighbor sample indices
  - ``neighbor_scores.npy``   shape ``(N, top_k)`` float32 — cosine similarities

Both arrays are opened in read-only ``mmap_mode="r"`` so multiple DataLoader
worker processes can share them without copying.

Usage
-----
    idx_path   = "outputs/imagenet100_b3/embeddings/neighbors.npy"
    score_path = "outputs/imagenet100_b3/embeddings/neighbor_scores.npy"

    ni = NeighborIndex(idx_path, score_path)

    # Top-30 positives for sample 42, filtered to cos-sim ≥ 0.5
    nbr_idx, nbr_sim = ni.get_positives(42, p=30, min_similarity=0.5)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class NeighborIndex:
    """Read-only, thread-safe access to precomputed nearest-neighbor rankings.

    Parameters
    ----------
    indices_path:
        Path to ``neighbors.npy``, shape ``(N, top_k)`` int32.
        ``indices[i, r]`` is the index of the r-th nearest neighbor of sample i.
    scores_path:
        Path to ``neighbor_scores.npy``, shape ``(N, top_k)`` float32.
        ``scores[i, r]`` is the cosine similarity between sample i and its
        r-th nearest neighbor (assumed to be sorted descending).
    """

    def __init__(self, indices_path: str, scores_path: str) -> None:
        if not os.path.exists(indices_path):
            raise FileNotFoundError(f"Neighbor indices file not found: {indices_path}")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Neighbor scores file not found: {scores_path}")

        # mmap_mode="r" → read-only; safe to share across processes/threads
        self._indices: np.ndarray = np.load(indices_path, mmap_mode="r")
        self._scores: np.ndarray = np.load(scores_path, mmap_mode="r")

        if self._indices.shape != self._scores.shape:
            raise ValueError(
                f"Shape mismatch: indices {self._indices.shape} vs "
                f"scores {self._scores.shape}"
            )
        if self._indices.ndim != 2:
            raise ValueError(
                f"Expected 2-D arrays (N, top_k), got shape {self._indices.shape}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        """Total number of samples in the index."""
        return self._indices.shape[0]

    @property
    def top_k(self) -> int:
        """Number of neighbors stored per sample."""
        return self._indices.shape[1]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_positives(
        self,
        idx: int,
        p: int = 30,
        min_similarity: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the positive-pool neighbor indices and scores for sample *idx*.

        The "positive pool" is ranks ``[0, p)`` — samples that are highly
        similar to the query but not identical.  An optional similarity
        threshold further filters out low-quality matches.

        Parameters
        ----------
        idx:
            Query sample index in ``[0, N)``.
        p:
            Upper rank bound (exclusive).  Returns at most *p* neighbors.
        min_similarity:
            Minimum cosine similarity threshold.  Neighbors below this value
            are excluded.  Set to ``0.0`` to disable filtering.

        Returns
        -------
        neighbor_indices : np.ndarray
            1-D int32 array of neighbor sample indices (length ≤ p).
        neighbor_scores : np.ndarray
            1-D float32 array of corresponding cosine similarities.

        Examples
        --------
        >>> ni = NeighborIndex("neighbors.npy", "scores.npy")
        >>> nbrs, sims = ni.get_positives(0, p=30, min_similarity=0.5)
        >>> len(nbrs) <= 30
        True
        """
        if not (0 <= idx < self.N):
            raise IndexError(f"idx={idx} out of range [0, {self.N})")
        if p <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        top_p = min(p, self.top_k)
        # Copy the slice out of the mmap'd array to a plain numpy array.
        # This is a small allocation (at most p ints/floats) and avoids
        # keeping a reference to the mmap page open longer than necessary.
        nbr_idx = np.array(self._indices[idx, :top_p], dtype=np.int32)
        nbr_sim = np.array(self._scores[idx, :top_p], dtype=np.float32)

        # Remove self-references (can occur if the pipeline didn't filter them)
        not_self = nbr_idx != idx
        nbr_idx = nbr_idx[not_self]
        nbr_sim = nbr_sim[not_self]

        if min_similarity > 0.0:
            mask = nbr_sim >= min_similarity
            nbr_idx = nbr_idx[mask]
            nbr_sim = nbr_sim[mask]

        return nbr_idx, nbr_sim

    def __repr__(self) -> str:
        return (
            f"NeighborIndex(N={self.N}, top_k={self.top_k}, "
            f"indices_dtype={self._indices.dtype}, "
            f"scores_dtype={self._scores.dtype})"
        )
