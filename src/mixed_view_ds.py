"""NeighborViewDataset — injects neighbor-local views into LeJEPA training.

Each sample produces:
  - V_global  global views   (224×224) from the anchor image
  - V_self    local  views   ( 96× 96) from the anchor image
  - V_neighbor local views   ( 96× 96) sampled from the positive-pool neighbors

The positive pool is defined by ``NeighborIndex.get_positives(idx, p=p)``
(ranks [0, p) in the CLIP similarity ranking).  This is the complement of the
"Goldilocks zone" [p, p+m) used for METIS batch mining.

``__getitem__`` returns a dict::

    {
        "global_views" : List[Tensor],   # length V_global
        "local_views"  : List[Tensor],   # length V_self + V_neighbor
        "label"        : int,
        "index"        : int,
    }

``collate_mixed_views`` converts a list of such dicts into the same
``(List[(B,C,H,W)], labels_Tensor)`` tuple that the existing ``collate_views``
and ``BaseTrainer.training_step`` expect — so this dataset is a drop-in
replacement for ``HFDataset`` with **zero** changes to the trainer logic.

Augmentation convention
-----------------------
CPU transforms are kept minimal (crop + flip + ``ToImage()``) matching
``HFDataset``.  Color jitter, Gaussian blur and normalization are applied by
the GPU augmentation hooks already present in ``BaseTrainer``.  No duplication.

Neighbor sampling modes
-----------------------
uniform  (default)  — uniform random draw from the positive pool
weighted            — draw proportional to softmax-normalised cosine similarities
top                 — always pick the top-V_neighbor most similar neighbors

Same-label neighbor pool (optional)
-----------------------------------
If ``neighbor_same_label_only=True``, the positive pool from ``get_positives``
(ranks ``[0, p)``) is intersected with images that share the anchor's class
label — a minimal "mixed + neighbor" scheme: teacher-ranked neighbors, but
only same-class instances (like ``CrossInstanceDataset`` mixed views, but
restricted to the top-*p* ranking).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torchvision.transforms import v2

from ds import HFDataset
from neighbor_index import NeighborIndex

logger = logging.getLogger(__name__)


class NeighborViewDataset(HFDataset):
    """HFDataset extended with teacher-guided neighbor local views.

    Parameters
    ----------
    split:
        ``"train"`` or ``"val"``.
    neighbor_index:
        Preloaded ``NeighborIndex`` wrapping neighbor_indices.npy + neighbor_scores.npy.
    V_global:
        Number of global (224×224) views from the anchor.
    V_self:
        Number of local (96×96) self-augmented views from the anchor.
    V_neighbor:
        Number of local (96×96) views sampled from positive-pool neighbors.
        Set to 0 to fall back to pure self-augmentation (equivalent to HFDataset).
    p:
        Positive-pool size: use ranks [0, p) from the neighbor index.
    min_similarity:
        Cosine similarity threshold.  Neighbors below this value are excluded
        from the pool.  Set to 0.0 to disable.
    neighbor_sampling:
        ``"uniform"`` (default) — uniform random draw from pool.
        ``"weighted"`` — draw proportional to softmax(similarities).
        ``"top"``      — always take the first V_neighbor entries in the pool.
    neighbor_start_epoch:
        Curriculum: for epochs < this value, use ``warmup_V_self`` self locals only
        (no neighbors); from this epoch onward use ``V_self`` self locals plus
        ``V_neighbor`` neighbor locals. Set to 0 to disable (neighbors from epoch 0).
    warmup_V_self:
        Number of self local crops during the warmup phase. If ``None`` and
        ``neighbor_start_epoch > 0`` and ``V_neighbor > 0``, defaults to
        ``V_self + V_neighbor`` so total local count matches the post-warmup phase
        (e.g. 6 self-only then 4 self + 2 neighbor).
    global_img_size:
        Spatial size of global crops (default 224).
    local_img_size:
        Spatial size of local crops (default 96).
    dataset:
        Dataset identifier passed to ``HFDataset._get_ds`` (e.g. ``"inet100"``).
    seed:
        Base random seed.  Varied per epoch via ``set_epoch()``.
    neighbor_same_label_only:
        If True, restrict the top-*p* neighbor pool to images with the same
        class label as the anchor (mixed-instance + teacher rank).
    """

    def __init__(
        self,
        split: str,
        neighbor_index: NeighborIndex,
        V_global: int = 2,
        V_self: int = 6,
        V_neighbor: int = 2,
        p: int = 30,
        min_similarity: float = 0.0,
        neighbor_sampling: Literal["uniform", "weighted", "top"] = "uniform",
        neighbor_start_epoch: int = 0,
        warmup_V_self: int | None = None,
        global_img_size: int = 224,
        local_img_size: int = 96,
        dataset: str = "inet100",
        seed: int = 0,
        neighbor_same_label_only: bool = False,
    ) -> None:
        # Initialise parent: loads the HF dataset + builds transforms
        super().__init__(
            split=split,
            V_global=V_global,
            V_local=V_self,            # parent's V_local ≡ our V_self
            global_img_size=global_img_size,
            local_img_size=local_img_size,
            dataset=dataset,
            seed=seed,
        )

        if neighbor_sampling not in ("uniform", "weighted", "top"):
            raise ValueError(
                f"neighbor_sampling must be 'uniform', 'weighted', or 'top', "
                f"got {neighbor_sampling!r}"
            )

        self.neighbor_index = neighbor_index
        self.V_self = V_self
        self.V_neighbor = V_neighbor
        self.p = p
        self.min_similarity = min_similarity
        self.neighbor_sampling = neighbor_sampling
        self.neighbor_start_epoch = neighbor_start_epoch
        self._epoch: int = 0
        self.neighbor_same_label_only = neighbor_same_label_only
        # Per-sample labels for same-class filtering (small vs. mmap'd images)
        self._labels: np.ndarray | None = None
        if self.neighbor_same_label_only:
            self._labels = np.asarray(self.ds["label"], dtype=np.int64)

        if neighbor_start_epoch > 0 and V_neighbor > 0:
            self.V_self_warmup = (
                warmup_V_self if warmup_V_self is not None else V_self + V_neighbor
            )
        else:
            self.V_self_warmup = V_self

        logger.info(
            "NeighborViewDataset: split=%s  N=%d  V_global=%d V_self=%d V_neighbor=%d  "
            "p=%d  min_sim=%.2f  sampling=%s  neighbor_start_epoch=%d  "
            "V_self_warmup=%d (epochs < start)  same_label_neighbors=%s",
            split, len(self.ds), V_global, V_self, V_neighbor,
            p, min_similarity, neighbor_sampling, neighbor_start_epoch,
            self.V_self_warmup,
            neighbor_same_label_only,
        )

    # ------------------------------------------------------------------
    # Epoch-level hook
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Vary the neighbor sampling seed each epoch for diversity."""
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Core item getter
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """Return a dict of views for sample *idx*.

        Returns
        -------
        dict with keys:
            ``"global_views"`` — List of V_global tensors, each (C, H, W)
            ``"local_views"``  — List of (V_self + V_neighbor) tensors, each (C, H, W)
            ``"label"``        — int class label
            ``"index"``        — int sample index (useful for analysis / debugging)
        """
        entry = self.ds[idx]
        img = self._load_image(entry)
        label: int = entry["label"]

        if self.split != "train":
            # Validation: delegate entirely to parent's test transform
            views, _ = super().__getitem__(idx)
            return {"global_views": views, "local_views": [], "label": label, "index": idx}

        # ── Global views from anchor ──────────────────────────────────
        global_views = [self.global_transform(img) for _ in range(self.V_global)]

        # ── Self local views (curriculum: more self-only during warmup) ─
        if (
            self.neighbor_start_epoch > 0
            and self.V_neighbor > 0
            and self._epoch < self.neighbor_start_epoch
        ):
            n_self = self.V_self_warmup
        else:
            n_self = self.V_self
        local_views = [self.local_transform(img) for _ in range(n_self)]

        # ── Neighbor local views (curriculum: delayed introduction) ────
        if self.V_neighbor > 0 and self._epoch >= self.neighbor_start_epoch:
            neighbor_views = self._sample_neighbor_views(idx, label)
            local_views.extend(neighbor_views)

        return {
            "global_views": global_views,
            "local_views": local_views,
            "label": label,
            "index": idx,
        }

    # ------------------------------------------------------------------
    # Neighbor sampling
    # ------------------------------------------------------------------

    def _sample_neighbor_views(self, idx: int, anchor_label: int) -> list[torch.Tensor]:
        """Sample V_neighbor local views from the positive pool of *idx*.

        Falls back to self-augmentation if the pool is empty.
        """
        nbr_idx, nbr_sim = self.neighbor_index.get_positives(
            idx, p=self.p, min_similarity=self.min_similarity
        )

        if len(nbr_idx) == 0:
            # No valid neighbors → fall back to self-augmentation
            return [self.local_transform(self._load_image(self.ds[idx]))
                    for _ in range(self.V_neighbor)]

        if self.neighbor_same_label_only and self._labels is not None:
            nbr_idx, nbr_sim = self._filter_same_label_pool(
                nbr_idx, nbr_sim, anchor_label
            )
        if len(nbr_idx) == 0:
            return [self.local_transform(self._load_image(self.ds[idx]))
                    for _ in range(self.V_neighbor)]

        # Per-sample reproducible RNG varied by epoch
        rng = np.random.default_rng(int(self.seed) + idx + self._epoch * 1_000_003)

        chosen_indices = self._choose_neighbors(rng, nbr_idx, nbr_sim)

        # Batch-fetch images from the HF dataset (single I/O call for multiple)
        entries = self.ds[chosen_indices.tolist()]

        if "image" in entries:
            imgs = entries["image"]
        elif "img" in entries:
            imgs = entries["img"]
        else:
            raise ValueError("Image key not found in neighbor entries")

        views = []
        for img in imgs:
            rgb = img if img.mode == "RGB" else img.convert("RGB")
            views.append(self.local_transform(rgb))
        return views

    def _filter_same_label_pool(
        self,
        nbr_idx: np.ndarray,
        nbr_sim: np.ndarray,
        anchor_label: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only neighbors in the top-*p* pool whose label matches the anchor."""
        assert self._labels is not None
        if len(nbr_idx) == 0:
            return nbr_idx, nbr_sim
        lbl = self._labels[nbr_idx.astype(np.int64)]
        mask = lbl == anchor_label
        return nbr_idx[mask], nbr_sim[mask]

    def _choose_neighbors(
        self,
        rng: np.random.Generator,
        nbr_idx: np.ndarray,
        nbr_sim: np.ndarray,
    ) -> np.ndarray:
        """Select V_neighbor indices from *nbr_idx* using the configured strategy."""
        n_avail = len(nbr_idx)
        n_pick = min(self.V_neighbor, n_avail)

        if self.neighbor_sampling == "top":
            chosen = nbr_idx[:n_pick]

        elif self.neighbor_sampling == "weighted":
            # Softmax over scores → probability distribution
            shifted = nbr_sim - nbr_sim.max()           # numerical stability
            weights = np.exp(shifted)
            weights /= weights.sum()
            chosen = rng.choice(nbr_idx, size=n_pick, replace=(n_avail < n_pick), p=weights)

        else:  # "uniform" (default)
            chosen = rng.choice(nbr_idx, size=n_pick, replace=(n_avail < n_pick))

        # If pool was smaller than V_neighbor, repeat to reach the target count
        if len(chosen) < self.V_neighbor:
            repeats = (self.V_neighbor + len(chosen) - 1) // len(chosen)
            chosen = np.tile(chosen, repeats)[: self.V_neighbor]

        return chosen

# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_mixed_views(
    batch: list[dict],
    include_index: bool = False,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate a list of ``NeighborViewDataset.__getitem__`` dicts.

    Produces the same ``(List[(B,C,H,W)], labels)`` tuple as ``collate_views``
    so it is a drop-in replacement for ``BaseTrainer.train_dataloader``'s
    ``collate_fn`` argument.

    Global views are placed first (large resolution), followed by all local
    views (self + neighbor), matching the resolution-sorting convention of
    ``collate_views``.

    Parameters
    ----------
    batch:
        List of dicts from ``NeighborViewDataset.__getitem__``.
    include_index:
        If True, the second element of the returned tuple becomes a dict
        ``{"labels": Tensor, "indices": Tensor}`` instead of a plain label
        tensor.  Set this for analysis / debugging only; the standard trainer
        expects a plain label tensor.

    Returns
    -------
    views : List[Tensor]
        Each element is ``(B, C, H, W)``.  Global views first, then locals.
    labels : Tensor or dict
        Shape ``(B,)`` int64, or dict if ``include_index=True``.
    """
    B = len(batch)

    # -- Stack global views ------------------------------------------------
    # All samples must have the same V_global; use the first as reference.
    n_global = len(batch[0]["global_views"])
    stacked_global: list[torch.Tensor] = []
    for v_idx in range(n_global):
        stacked_global.append(
            torch.stack([sample["global_views"][v_idx] for sample in batch])
        )

    # -- Stack local views -------------------------------------------------
    n_local = len(batch[0]["local_views"])
    stacked_local: list[torch.Tensor] = []
    for v_idx in range(n_local):
        stacked_local.append(
            torch.stack([sample["local_views"][v_idx] for sample in batch])
        )

    views = stacked_global + stacked_local

    # -- Labels ------------------------------------------------------------
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)

    if include_index:
        indices = torch.tensor([s["index"] for s in batch], dtype=torch.long)
        return views, {"labels": labels, "indices": indices}

    return views, labels
