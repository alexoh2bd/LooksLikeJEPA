#!/usr/bin/env python3
"""Analyze neighbor quality: label agreement and similarity for top-k neighbors.

For each sample, computes:
  - How many of the top n neighbors share the same label (same-class positives)
  - Mean/min/max cosine similarity of those neighbors

Expects:
  - neighbors.npy      shape (N, top_k) int32  — neighbor sample indices
  - neighbor_scores.npy shape (N, top_k) float32 — cosine similarities
  - metadata.json     list of {index, label} — from Stage 1 embeddings
    OR dataset labels loaded from HuggingFace (same order as embeddings)

Usage:
  uv run python scripts/analyze_neighbor_quality.py \\
    --neighbors_path data/b3/imagenet100_b3_P32B32/ranks/neighbors.npy \\
    --scores_path   data/b3/imagenet100_b3_P32B32/ranks/neighbor_scores.npy \\
    --metadata_path data/b3/imagenet100_b3_P32B32/embeddings/metadata.json \\
    --top_n 64

  # Or use ranks directory (auto-detect stitched or sharded files):
  uv run python scripts/analyze_neighbor_quality.py \\
    --ranks_dir data/b3/imagenet1k_b3/ranks \\
    --metadata_path ... --top_n 64 --plot --plot_dir plots/neighbor_analysis

  # Load labels from HuggingFace dataset (inet100/imagenet-1k):
  uv run python scripts/analyze_neighbor_quality.py \\
    --neighbors_path ... --scores_path ... \\
    --dataset inet100 --split train \\
    --top_n 64 --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Add src for potential dataset loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_neighbors_and_scores(
    neighbors_path: str | None,
    scores_path: str | None,
    ranks_dir: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load neighbor indices and scores from explicit paths or ranks directory.

    When ranks_dir is set, prefers stitched files (neighbors.npy, neighbor_scores.npy).
    If not found, loads and concatenates shards (neighbors_shard*_of*.npy, etc.).
    """
    if ranks_dir is not None:
        stitched_n = os.path.join(ranks_dir, "neighbors.npy")
        stitched_s = os.path.join(ranks_dir, "neighbor_scores.npy")
        if os.path.isfile(stitched_n) and os.path.isfile(stitched_s):
            return (
                np.load(stitched_n, mmap_mode="r"),
                np.load(stitched_s, mmap_mode="r"),
            )
        # Fall back to shards
        pattern = re.compile(r"neighbors_shard(\d+)_of(\d+)\.npy")

        def shard_key(p: str) -> tuple[int, int]:
            m = pattern.search(os.path.basename(p))
            return (int(m.group(2)), int(m.group(1))) if m else (0, 0)

        nbr_files = sorted(
            glob.glob(os.path.join(ranks_dir, "neighbors_shard*_of*.npy")),
            key=shard_key,
        )
        if not nbr_files:
            raise FileNotFoundError(
                f"No neighbors.npy or neighbors_shard*_of*.npy in {ranks_dir}"
            )
        m = pattern.search(os.path.basename(nbr_files[0]))
        if not m:
            raise ValueError(f"Could not parse shard pattern from {nbr_files[0]}")
        num_shards = int(m.group(2))
        score_files = [
            os.path.join(ranks_dir, f"neighbors_scores_shard{sid}_of{num_shards}.npy")
            for sid in range(num_shards)
        ]
        if not all(os.path.isfile(f) for f in score_files):
            raise FileNotFoundError(
                f"Missing score shards in {ranks_dir}; expected {score_files}"
            )
        shards_n = [np.load(f, mmap_mode="r") for f in nbr_files]
        shards_s = [np.load(f, mmap_mode="r") for f in score_files]
        return np.concatenate(shards_n, axis=0), np.concatenate(shards_s, axis=0)

    if neighbors_path is None or scores_path is None:
        raise ValueError("Provide --neighbors_path and --scores_path, or --ranks_dir")
    return (
        np.load(neighbors_path, mmap_mode="r"),
        np.load(scores_path, mmap_mode="r"),
    )


def load_labels_from_metadata(path: str) -> np.ndarray:
    """Load labels from metadata.json produced by Stage 1."""
    with open(path) as f:
        meta = json.load(f)
    # Ensure sorted by index
    meta_sorted = sorted(meta, key=lambda x: x["index"])
    return np.array([m["label"] for m in meta_sorted], dtype=np.int64)


def load_labels_from_dataset(
    dataset: str, split: str, project_root: str | None = None
) -> np.ndarray:
    """Load labels from HuggingFace dataset (same order as B3 embeddings)."""
    from datasets import load_dataset

    root = project_root or os.path.join(os.path.dirname(__file__), "..")

    if dataset == "inet100":
        inet_dir = os.path.join(
            root,
            "data/cache/datasets--clane9--imagenet-100/snapshots/"
            "0519dc2f402a3a18c6e57f7913db059215eee25b/data/",
        )
        filenames = {"train": inet_dir + "train-*.parquet", "val": inet_dir + "validation*.parquet"}
        ds = load_dataset("parquet", data_files=filenames, split=split)
    elif dataset == "imagenet-1k":
        inet_dir = os.path.join(
            root,
            "data/hub/datasets--ILSVRC--imagenet-1k/snapshots/"
            "49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data",
        )
        filenames = {"train": inet_dir + "/train*.parquet", "val": inet_dir + "/validation*.parquet"}
        ds = load_dataset("parquet", data_files=filenames, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return np.array(ds["label"], dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze neighbor quality: label agreement and similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--neighbors_path",
        default=None,
        help="Path to neighbors.npy (omit if using --ranks_dir)",
    )
    p.add_argument(
        "--scores_path",
        default=None,
        help="Path to neighbor_scores.npy (omit if using --ranks_dir)",
    )
    p.add_argument(
        "--ranks_dir",
        default=None,
        help="Path to ranks directory (auto-detect stitched or sharded files)",
    )
    p.add_argument(
        "--metadata_path",
        default=None,
        help="Path to metadata.json (from Stage 1). Overrides --dataset.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (inet100, imagenet-1k) to load labels from HF.",
    )
    p.add_argument("--split", default="train", help="Dataset split when using --dataset")
    p.add_argument(
        "--top_n",
        type=int,
        default=64,
        help="Number of top neighbors to analyze (e.g. top 64)",
    )
    p.add_argument(
        "--training_window",
        type=int,
        default=16,
        help="Top-k window used in training (kP). Per-class purity uses this.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, analyze only a random subsample of N queries (for large datasets)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--project_root",
        default=None,
        help="Project root for dataset paths (default: parent of scripts/)",
    )
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument(
        "--plot_dir",
        default="./plots",
        help="Output directory for plots (when --plot)",
    )
    args = p.parse_args()

    if args.ranks_dir is None and (args.neighbors_path is None or args.scores_path is None):
        p.error("Provide either --ranks_dir or both --neighbors_path and --scores_path")

    # Load neighbor data
    indices, scores = load_neighbors_and_scores(
        args.neighbors_path, args.scores_path, args.ranks_dir
    )

    if indices.shape != scores.shape:
        raise ValueError(
            f"Shape mismatch: indices {indices.shape} vs scores {scores.shape}"
        )

    N, top_k = indices.shape
    if args.top_n > top_k:
        print(f"Warning: top_n={args.top_n} > top_k={top_k}. Using top_k.")
        top_n = top_k
    else:
        top_n = args.top_n

    # Load labels
    if args.metadata_path:
        labels = load_labels_from_metadata(args.metadata_path)
    elif args.dataset:
        labels = load_labels_from_dataset(
            args.dataset, args.split, project_root=args.project_root
        )
    else:
        raise ValueError("Provide either --metadata_path or --dataset")

    if len(labels) != N:
        raise ValueError(
            f"Label count {len(labels)} != neighbor matrix rows {N}"
        )

    # Optionally subsample queries
    rng = np.random.default_rng(args.seed)
    if args.sample is not None and args.sample < N:
        query_indices = rng.choice(N, size=args.sample, replace=False)
    else:
        query_indices = np.arange(N)

    # Compute per-query stats
    same_label_counts = []
    mean_sims = []
    min_sims = []
    max_sims = []

    for i in query_indices:
        query_label = labels[i]
        nbr_idx = indices[i, :top_n]
        nbr_sim = scores[i, :top_n]

        # Exclude self (if present)
        mask = nbr_idx != i
        nbr_idx = nbr_idx[mask]
        nbr_sim = nbr_sim[mask]

        nbr_labels = labels[nbr_idx]
        same = (nbr_labels == query_label).astype(np.int32)
        same_count = int(same.sum())
        same_label_counts.append(same_count)

        if len(nbr_sim) > 0:
            mean_sims.append(float(nbr_sim.mean()))
            min_sims.append(float(nbr_sim.min()))
            max_sims.append(float(nbr_sim.max()))
        else:
            mean_sims.append(np.nan)
            min_sims.append(np.nan)
            max_sims.append(np.nan)

    same_label_counts = np.array(same_label_counts)
    mean_sims = np.array(mean_sims)
    min_sims = np.array(min_sims)
    max_sims = np.array(max_sims)

    # Average distributions at P=1, 5, 10, 20, 32, 48
    P_VALUES = [1, 5, 10, 20, 32, 48]
    p_stats = []
    for p in P_VALUES:
        if p > top_k:
            continue
        slc_p = []
        mean_sim_p = []
        for i in query_indices:
            query_label = labels[i]
            nbr_idx = indices[i, :p]
            nbr_sim = scores[i, :p]
            mask = nbr_idx != i
            nbr_idx = nbr_idx[mask]
            nbr_sim = nbr_sim[mask]
            nbr_labels = labels[nbr_idx]
            same = (nbr_labels == query_label).astype(np.int32)
            slc_p.append(int(same.sum()))
            if len(nbr_sim) > 0:
                mean_sim_p.append(float(nbr_sim.mean()))
            else:
                mean_sim_p.append(np.nan)
        slc_p = np.array(slc_p)
        mean_sim_p = np.array(mean_sim_p)
        p_stats.append({
            "P": p,
            "mean_same_label": slc_p.mean(),
            "std_same_label": slc_p.std(),
            "mean_sim": np.nanmean(mean_sim_p),
        })

    # Per-image purity for top-kP only (actual training window)
    kP = min(args.training_window, top_k)
    # neighbors[:, :kP] -> (N, kP), labels[:, None] -> (N, 1), broadcast -> (N, kP)
    topkP_purity = (labels[indices[:, :kP]] == labels[:, None]).mean(axis=1)
    frac_images_low_purity = (topkP_purity < 0.5).mean()

    # Per-class purity: mean purity over images in each class
    unique_labels = np.unique(labels)
    class_purities = np.array([topkP_purity[labels == c].mean() for c in unique_labels])
    frac_classes_low_purity = (class_purities < 0.5).mean() if len(class_purities) > 0 else 0.0

    # Report
    data_src = args.ranks_dir or args.neighbors_path or "(unknown)"
    print("=" * 60)
    print("Neighbor Quality Analysis")
    print("=" * 60)
    print(f"Data source:    {data_src}")
    print(f"Top-n:          {top_n}")
    print(f"Queries:        {len(query_indices)} (of {N} total)")
    print()
    print("--- Label agreement (same-class in top-n) ---")
    print(f"  Mean same-label count:  {same_label_counts.mean():.2f} / {top_n}")
    print(f"  Std:                    {same_label_counts.std():.2f}")
    print(f"  Min:                    {same_label_counts.min()}")
    print(f"  Max:                    {same_label_counts.max()}")
    print(f"  Median:                 {np.median(same_label_counts):.0f}")
    print()
    print("--- Cosine similarity (top-n neighbors) ---")
    print(f"  Mean similarity:        {np.nanmean(mean_sims):.4f}")
    print(f"  Min (across neighbors): {np.nanmean(min_sims):.4f}")
    print(f"  Max (across neighbors): {np.nanmean(max_sims):.4f}")
    print()
    print("--- Percentiles (same-label count) ---")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  {pct}th: {np.percentile(same_label_counts, pct):.0f}")
    print()
    print("--- Average distributions at P=1, 5, 10, 20, 32, 48 ---")
    for s in p_stats:
        print(f"  P={s['P']:2d}:  mean same-label {s['mean_same_label']:.2f} / {s['P']}  "
              f"(std {s['std_same_label']:.2f})   mean cosine sim {s['mean_sim']:.4f}")
    print()
    print("--- Top-kP purity (training window kP={}) ---".format(kP))
    print(f"  Per-image purity:  mean {topkP_purity.mean():.3f}  median {np.median(topkP_purity):.3f}")
    print(f"  Fraction of images with purity < 0.5:  {frac_images_low_purity:.1%}")
    if frac_images_low_purity > 0.2:
        print("  *** CONTAMINATION SIGNIFICANT: >20% of images have top-{} purity < 0.5 ***".format(kP))
    print(f"  Per-class purity:  mean {class_purities.mean():.3f}  median {np.median(class_purities):.3f}")
    print(f"  Fraction of classes with mean purity < 0.5:  {frac_classes_low_purity:.1%}")
    if frac_classes_low_purity > 0:
        n_low = (class_purities < 0.5).sum()
        print(f"  Classes with <50% purity (exclude or down-weight): {n_low} / {len(class_purities)}")
    print("=" * 60)

    # Plots
    if args.plot:
        _generate_plots(
            indices=indices,
            scores=scores,
            labels=labels,
            query_indices=query_indices,
            top_n=top_n,
            topkP_purity=topkP_purity,
            class_purities=class_purities,
            training_window=kP,
            plot_dir=args.plot_dir,
            rng=rng,
        )


def _generate_plots(
    indices: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    query_indices: np.ndarray,
    top_n: int,
    topkP_purity: np.ndarray,
    class_purities: np.ndarray,
    training_window: int,
    plot_dir: str,
    rng: np.random.Generator,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # Subsample for histogram if too many scores (cap at ~500k)
    max_hist_samples = 500_000
    n_queries = len(query_indices)
    total_scores = n_queries * top_n
    if total_scores > max_hist_samples:
        q_subset = rng.choice(query_indices, size=max_hist_samples // top_n, replace=False)
    else:
        q_subset = query_indices

    scores_flat = scores[np.ix_(q_subset, np.arange(top_n))].flatten()
    # Exclude self (score 1.0) if present
    scores_flat = scores_flat[scores_flat < 1.0 - 1e-6]

    # 1. Scores histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_flat, bins=80, density=True, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Cosine similarity (top-n neighbors)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of neighbor cosine similarities")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "scores_histogram.png"), dpi=150)
    plt.close(fig)

    # 2. Same-label count distribution (exclude self like main analysis)
    same_label_counts = []
    for i in query_indices:
        nbr_idx = indices[i, :top_n]
        mask = nbr_idx != i
        nbr_idx = nbr_idx[mask]
        nbr_labels = labels[nbr_idx]
        same_label_counts.append((nbr_labels == labels[i]).sum())
    same_label_counts = np.array(same_label_counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        same_label_counts,
        bins=np.arange(-0.5, top_n + 1.5, 1),
        density=True,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Same-label count in top-n")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of same-class neighbors (top-{top_n})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "same_label_dist.png"), dpi=150)
    plt.close(fig)

    # 2b. Per-image top-kP purity histogram (training window)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(topkP_purity, bins=50, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", label="50% threshold")
    ax.set_xlabel("Same-label fraction in top-{} neighbors".format(training_window))
    ax.set_ylabel("Count")
    ax.set_title("Per-image purity (top-{} training window)".format(training_window))
    frac_low = (topkP_purity < 0.5).mean()
    ax.text(0.02, 0.98, f"{frac_low:.1%} of images have purity < 0.5", transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "topkP_purity_histogram.png"), dpi=150)
    plt.close(fig)

    # 2c. Per-class purity histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(class_purities, bins=50, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", label="50% threshold")
    ax.set_xlabel("Mean same-label fraction in top-{} neighbors (per class)".format(training_window))
    ax.set_ylabel("Number of classes")
    ax.set_title("Per-class neighbor purity (top-{} training window)".format(training_window))
    frac_classes_low = (class_purities < 0.5).mean()
    n_low = (class_purities < 0.5).sum()
    ax.text(0.02, 0.98, f"{n_low} classes ({frac_classes_low:.1%}) have purity < 0.5", transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "per_class_purity_histogram.png"), dpi=150)
    plt.close(fig)

    # 3. Similarity by label (same vs different)
    sim_same = []
    sim_diff = []
    for i in query_indices:
        nbr_idx = indices[i, :top_n]
        nbr_sim = scores[i, :top_n]
        mask_self = nbr_idx != i
        nbr_idx = nbr_idx[mask_self]
        nbr_sim = nbr_sim[mask_self]
        nbr_labels = labels[nbr_idx]
        same_mask = nbr_labels == labels[i]
        if same_mask.sum() > 0:
            sim_same.extend(nbr_sim[same_mask].tolist())
        if (~same_mask).sum() > 0:
            sim_diff.extend(nbr_sim[~same_mask].tolist())

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        [sim_same, sim_diff],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same label", "Different label"])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Neighbor similarity: same-class vs different-class")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "similarity_by_label.png"), dpi=150)
    plt.close(fig)

    # 4. Per-rank analysis
    mean_sim_per_rank = np.array(
        [scores[query_indices, r].mean() for r in range(top_n)]
    )
    same_label_frac_per_rank = np.array(
        [
            (labels[indices[query_indices, r]] == labels[query_indices]).mean()
            for r in range(top_n)
        ]
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(np.arange(1, top_n + 1), mean_sim_per_rank, "b-", label="Mean cosine sim")
    ax1.set_xlabel("Rank (1 = nearest)")
    ax1.set_ylabel("Mean cosine similarity", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, top_n + 1), same_label_frac_per_rank, "r-", label="Same-label fraction")
    ax2.set_ylabel("Same-label fraction", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("Per-rank: similarity and same-label fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "per_rank_analysis.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
