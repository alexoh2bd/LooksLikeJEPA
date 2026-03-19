"""Stage 5 — Export batch schedules and finalise the LeJEPABatchSampler.

Reads ``clusters.json`` (Stage 4) and:

1. **Validates** that the cluster file is consistent with the requested
   ``batch_size`` and ``K``.
2. **Generates static epoch batch files** (optional, ``--export_static``):
   each file ``epoch_{n}_batches.npy`` is a ``(num_batches, batch_size)``
   int64 array of sample indices — useful for pre-computing the full schedule
   or for debugging.
3. **Copies ``batch_sampler.py``** to ``<output_dir>/batch_sampler.py`` so
   users can import it from the output directory directly.
4. Prints an integration snippet showing how to wire the sampler into a
   PyTorch / Lightning DataLoader.

Usage
-----
    python 05_export_batches.py \\
        --clusters_path outputs/imagenet1k_b3/clusters/clusters.json \\
        --output_dir    outputs/imagenet1k_b3/batches \\
        --batch_size 256 --K 32 --num_epochs 100

    # Also export static per-epoch schedule files:
    python 05_export_batches.py ... --export_static
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import Timer, ensure_dir, set_seed, setup_logging

# Locate the batch_sampler module relative to this file (one level up)
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PIPELINE_ROOT)

logger = setup_logging("05_export_batches")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_clusters(
    clusters: dict[str, list[int]],
    K: int,
    batch_size: int,
) -> dict:
    """Validate cluster file and return summary statistics."""
    if batch_size % K != 0:
        raise ValueError(
            f"batch_size={batch_size} is not divisible by K={K}."
        )

    sizes = [len(v) for v in clusters.values()]
    n_clusters = len(clusters)
    total_samples = sum(sizes)
    size_histogram = {}
    for s in sizes:
        size_histogram[s] = size_histogram.get(s, 0) + 1

    non_K = sum(1 for s in sizes if s != K)
    if non_K > 0:
        logger.warning(
            "%d / %d clusters do not have size K=%d. "
            "This may cause the last batch of some epochs to be slightly smaller.",
            non_K, n_clusters, K,
        )

    clusters_per_batch = batch_size // K
    batches_per_epoch = n_clusters // clusters_per_batch

    stats = {
        "n_clusters": n_clusters,
        "K": K,
        "batch_size": batch_size,
        "clusters_per_batch": clusters_per_batch,
        "batches_per_epoch": batches_per_epoch,
        "total_samples": total_samples,
        "samples_per_epoch": batches_per_epoch * batch_size,
        "coverage_pct": (batches_per_epoch * batch_size) / total_samples * 100,
        "size_histogram": size_histogram,
    }
    logger.info("Cluster validation stats:\n%s", json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Static schedule export
# ---------------------------------------------------------------------------

def export_static_schedules(
    clusters: dict[str, list[int]],
    output_dir: str,
    batch_size: int,
    K: int,
    num_epochs: int,
    seed: int,
) -> None:
    """Write per-epoch batch index files to *output_dir*.

    Each file ``epoch_{n}_batches.npy`` is a 2-D int64 array of shape
    ``(num_batches, batch_size)`` where each row is one batch of sample
    indices.
    """
    import random

    clusters_per_batch = batch_size // K
    cluster_ids = sorted(clusters.keys(), key=int)
    n_clusters = len(cluster_ids)

    logger.info(
        "Exporting static schedules: %d epochs × ~%d batches/epoch",
        num_epochs, n_clusters // clusters_per_batch,
    )

    for epoch in range(num_epochs):
        rng = random.Random(seed + epoch * 31337)
        order = list(cluster_ids)
        rng.shuffle(order)

        batches: list[list[int]] = []
        for start in range(0, n_clusters - clusters_per_batch + 1, clusters_per_batch):
            batch_cluster_ids = order[start: start + clusters_per_batch]
            indices: list[int] = []
            for cid in batch_cluster_ids:
                cluster = clusters[cid]
                if len(cluster) > K:
                    chosen = rng.sample(range(len(cluster)), K)
                    indices.extend(cluster[i] for i in chosen)
                else:
                    indices.extend(cluster)
            batches.append(indices)

        if not batches:
            logger.warning("Epoch %d: no complete batches generated.", epoch)
            continue

        arr = np.array(batches, dtype=np.int64)
        out_path = os.path.join(output_dir, f"epoch_{epoch}_batches.npy")
        np.save(out_path, arr)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.info("  epoch %d: %s  shape=%s", epoch, out_path, arr.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # ------------------------------------------------------------------
    # Load clusters
    # ------------------------------------------------------------------
    with Timer("Loading clusters.json", logger):
        with open(args.clusters_path) as fh:
            clusters: dict[str, list[int]] = json.load(fh)
    logger.info("Loaded %d clusters from %s", len(clusters), args.clusters_path)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    stats = validate_clusters(clusters, args.K, args.batch_size)

    # Save stats
    stats_path = os.path.join(args.output_dir, "batch_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Batch stats saved to %s", stats_path)

    # ------------------------------------------------------------------
    # Copy batch_sampler.py to output dir
    # ------------------------------------------------------------------
    src_sampler = os.path.join(_PIPELINE_ROOT, "batch_sampler.py")
    dst_sampler = os.path.join(args.output_dir, "batch_sampler.py")
    if os.path.exists(src_sampler):
        shutil.copy2(src_sampler, dst_sampler)
        logger.info("Copied batch_sampler.py → %s", dst_sampler)
    else:
        logger.warning("Could not find batch_sampler.py at %s", src_sampler)

    # ------------------------------------------------------------------
    # Optional: export static per-epoch schedule files
    # ------------------------------------------------------------------
    if args.export_static:
        with Timer("Exporting static epoch schedules", logger):
            export_static_schedules(
                clusters=clusters,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                K=args.K,
                num_epochs=args.num_epochs,
                seed=args.seed,
            )

    # ------------------------------------------------------------------
    # Print integration snippet
    # ------------------------------------------------------------------
    snippet = f"""
# ─── Integration snippet ──────────────────────────────────────────────────
import sys
sys.path.insert(0, "{_PIPELINE_ROOT}")
from batch_sampler import LeJEPABatchSampler
from torch.utils.data import DataLoader

sampler = LeJEPABatchSampler(
    clusters_path="{args.clusters_path}",
    batch_size={args.batch_size},
    K={args.K},
    seed={args.seed},
    epoch=0,          # call sampler.set_epoch(epoch) each training epoch
)

loader = DataLoader(
    train_dataset,    # your ImageNet Dataset instance
    batch_sampler=sampler,
    num_workers=8,
    pin_memory=True,
)
# ─────────────────────────────────────────────────────────────────────────
"""
    logger.info(snippet)

    logger.info("Stage 5 complete. Output directory: %s", args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 5: Export batch schedules and LeJEPABatchSampler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Optional path to default.yaml.")
    p.add_argument(
        "--clusters_path",
        default="outputs/imagenet1k_b3/clusters/clusters.json",
        help="Path to clusters.json from Stage 4.",
    )
    p.add_argument("--output_dir", default="outputs/imagenet1k_b3/batches",
                   help="Directory for batch schedule outputs.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--K", type=int, default=32,
                   help="Samples per cluster (must match Stage 4).")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export_static", action="store_true",
                   help="Write per-epoch batch index arrays to disk.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        from utils import load_yaml_config
        cfg = load_yaml_config(args.config)
        b = cfg.get("batching", {})
        if args.batch_size == 256:
            args.batch_size = b.get("batch_size", args.batch_size)
        if args.K == 32:
            args.K = cfg.get("clustering", {}).get("K", args.K)
        if args.num_epochs == 100:
            args.num_epochs = b.get("num_epochs", args.num_epochs)
        if args.seed == 42:
            args.seed = b.get("seed", args.seed)

    logger.info("Configuration: %s", vars(args))
    main(args)
