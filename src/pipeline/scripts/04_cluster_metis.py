"""Stage 4 — METIS community detection to form semantically-coherent clusters.

Reads ``adjacency_list.pkl`` (Stage 3) and runs ``pymetis.part_graph`` to
partition the graph into clusters of target size ``K``.

METIS minimises edge cuts, which means samples that are mutual hard negatives
(connected in the graph) end up in the **same** cluster — exactly the property
we need for JEPA batch mining.

Post-processing
---------------
METIS does not guarantee equal-sized partitions.  We apply the same
fill-and-trim logic as ``src/clustering.py`` (lines 283–340) to produce
clusters of exactly ``K`` samples.

Outputs
-------
  <output_dir>/cluster_assignments.npy  — int32, shape (N,): cluster ID per sample
  <output_dir>/clusters.json            — dict {str(cluster_id): [idx, ...]}
  <output_dir>/cluster_stats.json       — size histogram and summary

Usage
-----
    python 04_cluster_metis.py \\
        --graph_path outputs/imagenet1k_b3/graph/adjacency_list.pkl \\
        --output_dir outputs/imagenet1k_b3/clusters \\
        --K 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import Timer, ensure_dir, set_seed, setup_logging

logger = setup_logging("04_cluster_metis")


# ---------------------------------------------------------------------------
# METIS partitioning
# ---------------------------------------------------------------------------

def metis_partition(adjacency_list: list[list[int]], n_clusters: int) -> list[int]:
    """Partition *adjacency_list* into *n_clusters* parts using pymetis.

    Parameters
    ----------
    adjacency_list:
        ``adjacency_list[i]`` is a list of integer neighbour indices for node i.
        Must be undirected (if ``j`` is in ``adjacency_list[i]``, then ``i``
        must be in ``adjacency_list[j]``).  Self-loops are not allowed.
    n_clusters:
        Target number of partitions.

    Returns
    -------
    list[int]
        Partition assignment for each node (length N).
    """
    import pymetis

    N = len(adjacency_list)
    logger.info("Running pymetis.part_graph: N=%d, n_clusters=%d", N, n_clusters)

    # pymetis expects a flat adjacency structure:
    # xadj[i], xadj[i+1] mark the start/end of adjacency_list[i] in adjncy.
    xadj = [0]
    adjncy: list[int] = []
    for neighbours in adjacency_list:
        adjncy.extend(neighbours)
        xadj.append(len(adjncy))

    with Timer("pymetis.part_graph", logger):
        edgecuts, membership = pymetis.part_graph(
            n_clusters,
            xadj=xadj,
            adjncy=adjncy,
        )

    logger.info("METIS edgecuts: %d", edgecuts)
    return list(membership)


# ---------------------------------------------------------------------------
# Cluster resizing (fill/trim to exactly K)
# ---------------------------------------------------------------------------

def resize_clusters(
    raw_clusters: dict[int, list[int]],
    K: int,
    seed: int,
) -> list[list[int]]:
    """Resize METIS output clusters so every cluster has exactly K samples.

    Strategy (mirrors ``src/clustering.py``):
    1. Sort clusters by size (largest first).
    2. Pop excess samples from oversized clusters into a pool.
    3. Fill undersized clusters from the pool.
    4. Stitch any remaining pool samples into a final cluster; if it's smaller
       than K, borrow one sample from each prior cluster until full.

    Parameters
    ----------
    raw_clusters:
        Dict mapping cluster_id → list of sample indices.
    K:
        Target cluster size.
    seed:
        Random seed for reproducibility when choosing samples to borrow.
    """
    random.seed(seed)

    clusters_by_size = sorted(raw_clusters.values(), key=len, reverse=True)
    # Work on mutable copies
    clusters_by_size = [list(c) for c in clusters_by_size]

    final_clusters: list[list[int]] = []
    remaining: list[int] = []

    for cluster in clusters_by_size:
        while len(cluster) > K:
            remaining.append(cluster.pop())
        while len(cluster) < K and remaining:
            cluster.append(remaining.pop())
        final_clusters.append(cluster)

    final_clusters.sort(key=lambda x: min(x))

    if remaining:
        final_clusters.append(remaining)

    # Handle last cluster if it is not exactly K
    if len(final_clusters[-1]) != K:
        leftover = final_clusters.pop()
        # Split leftover into K-sized chunks
        chunks = [leftover[i: i + K] for i in range(0, len(leftover), K)]
        if len(chunks[-1]) < K:
            short_chunk = chunks.pop()
            needed = K - len(short_chunk)
            # Borrow one sample from prior clusters (round-robin)
            for c in final_clusters:
                if needed == 0:
                    break
                idx = random.randrange(len(c))
                short_chunk.append(c[idx])
                needed -= 1
            chunks.append(short_chunk)
        final_clusters.extend(chunks)

    n_total = sum(len(c) for c in final_clusters)
    assert n_total >= len(sum(raw_clusters.values(), [])), "Lost samples during resize."
    non_K = [len(c) for c in final_clusters if len(c) != K]
    if non_K:
        logger.warning("%d cluster(s) do not have size K=%d: %s", len(non_K), K, non_K[:10])

    return final_clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # ------------------------------------------------------------------
    # Load adjacency list
    # ------------------------------------------------------------------
    with Timer("Loading adjacency_list.pkl", logger):
        with open(args.graph_path, "rb") as fh:
            adj: list[list[int]] = pickle.load(fh)
    N = len(adj)
    logger.info("Adjacency list: %d nodes", N)

    # ------------------------------------------------------------------
    # Determine number of clusters
    # ------------------------------------------------------------------
    n_clusters = N // args.K
    remainder = N % args.K
    logger.info(
        "Target: K=%d  →  n_clusters=%d  (remainder=%d samples will be absorbed)",
        args.K, n_clusters, remainder,
    )
    if n_clusters < 2:
        raise ValueError(
            f"Too few samples (N={N}) for K={args.K}: would produce only {n_clusters} cluster(s)."
        )

    # ------------------------------------------------------------------
    # METIS partitioning
    # ------------------------------------------------------------------
    with Timer("METIS partitioning", logger):
        membership = metis_partition(adj, n_clusters)

    # Group into clusters
    raw_clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        raw_clusters[cid].append(idx)

    raw_sizes = Counter(len(v) for v in raw_clusters.values())
    logger.info("Raw cluster size distribution (top 10): %s",
                raw_sizes.most_common(10))

    # ------------------------------------------------------------------
    # Resize clusters to exactly K
    # ------------------------------------------------------------------
    with Timer("Resizing clusters to K", logger):
        final_clusters = resize_clusters(raw_clusters, args.K, args.seed)

    size_counts = Counter(len(c) for c in final_clusters)
    logger.info("Final cluster sizes: %s", dict(size_counts))
    logger.info("Number of final clusters: %d", len(final_clusters))

    # ------------------------------------------------------------------
    # Build output arrays
    # ------------------------------------------------------------------
    cluster_assignments = np.full(N, -1, dtype=np.int32)
    clusters_dict: dict[str, list[int]] = {}

    for cid, cluster in enumerate(tqdm(final_clusters, desc="Building output")):
        clusters_dict[str(cid)] = cluster
        for idx in cluster:
            if idx < N:
                cluster_assignments[idx] = cid

    unassigned = int((cluster_assignments == -1).sum())
    if unassigned > 0:
        logger.warning("%d samples have no cluster assignment (may be borrowed duplicates).", unassigned)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    asgn_path = os.path.join(args.output_dir, "cluster_assignments.npy")
    np.save(asgn_path, cluster_assignments)
    logger.info("Saved cluster_assignments.npy → %s", asgn_path)

    clusters_path = os.path.join(args.output_dir, "clusters.json")
    with open(clusters_path, "w") as fh:
        json.dump(clusters_dict, fh)
    logger.info("Saved clusters.json → %s  (%d clusters)", clusters_path, len(clusters_dict))

    # Cluster stats
    stats = {
        "N": N,
        "K": args.K,
        "n_clusters": len(final_clusters),
        "size_distribution": {str(k): v for k, v in Counter(len(c) for c in final_clusters).items()},
        "unassigned_samples": unassigned,
    }
    stats_path = os.path.join(args.output_dir, "cluster_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Cluster stats: %s", json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4: METIS graph partitioning into K-sized clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Optional path to default.yaml.")
    p.add_argument(
        "--graph_path",
        default="outputs/imagenet1k_b3/graph/adjacency_list.pkl",
        help="Path to adjacency_list.pkl from Stage 3.",
    )
    p.add_argument("--output_dir", default="outputs/imagenet1k_b3/clusters",
                   help="Directory for cluster output files.")
    p.add_argument("--K", type=int, default=32,
                   help="Target cluster size (samples per cluster).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        from utils import load_yaml_config
        cfg = load_yaml_config(args.config)
        c = cfg.get("clustering", {})
        if args.K == 32:
            args.K = c.get("K", args.K)
        if args.seed == 42:
            args.seed = c.get("seed", args.seed)

    logger.info("Configuration: %s", vars(args))
    main(args)
