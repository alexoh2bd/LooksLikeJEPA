"""Stage 3 — Construct a sparse adjacency graph from the rank matrix.

Reads ``neighbors.npy`` (shape ``(N, top_k)``) and builds a bidirectional
adjacency list suitable for METIS partitioning (Stage 4).

The B3 "Goldilocks zone" logic
-------------------------------
For each query ``i`` keep neighbors at ranks ``[p, p+m)``::

  ranks <  p   →  too similar   (likely false negatives / same class)
  ranks ≥ p+m  →  too easy      (not hard enough as negatives)

An edge ``(i, j)`` is only kept when **mutual**: ``j`` must also list ``i``
in its ``[p, p+m)`` window.

Memory-efficient implementation
---------------------------------
The naive approach of storing 128M ``(int, int)`` tuples in a Python ``set``
costs ~8 GB of RAM.  Instead we:

1. Extract the window slice: ``neighbors[:, p:p+m]``  shape ``(N, m)``
2. Encode every directed edge ``(i, j)`` as a single ``int64`` scalar
   ``i * N + j``  (fits since N ≤ 1.28M, so max value ≈ 1.64 × 10¹²).
3. Sort the ``(N*m,)`` int64 array  — ~1 GB.
4. For each edge check whether the reverse ``j * N + i`` exists using
   ``np.searchsorted`` — O(N·m·log(N·m)) but fully vectorised.

Parallel sharding
-----------------
Pass ``--num_shards S --shard_id i`` to build the **directed** edges for a
row slice only, writing ``directed_edges_shard{i}_of{S}.npy``.
A subsequent merge step (``--merge``) loads all shard edge arrays, combines
them, and performs the bidirectionality + label filter to produce the final
``adjacency_list.pkl``.

Outputs
-------
  <output_dir>/adjacency_list.pkl   — list[list[int]], length N
  <output_dir>/graph_stats.json     — summary statistics

Usage
-----
    # Single job
    python 03_build_sparse_graph.py \\
        --neighbors_path outputs/.../ranks/neighbors.npy \\
        --output_dir outputs/.../graph --p 30 --m 100

    # SLURM array (4 shards)
    python 03_build_sparse_graph.py ... --num_shards 4 --shard_id $SLURM_ARRAY_TASK_ID

    # Merge after array
    python 03_build_sparse_graph.py ... --merge --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import Timer, ensure_dir, setup_logging

logger = setup_logging("03_build_sparse_graph")


# ---------------------------------------------------------------------------
# Directed edge extraction (one shard)
# ---------------------------------------------------------------------------

def extract_directed_edges(
    neighbors: np.ndarray,
    p: int,
    m: int,
    shard_start: int,
    shard_end: int,
) -> np.ndarray:
    """Return all directed edges ``(i, j)`` for rows ``[shard_start, shard_end)``.

    Edges are returned as a ``(K, 2)`` int32 array where ``K ≤ shard_size * m``.
    Self-loops are excluded.
    """
    window = neighbors[shard_start:shard_end, p: p + m]  # (S, m)
    S, m_width = window.shape
    rows = np.repeat(np.arange(shard_start, shard_end, dtype=np.int32), m_width)
    cols = window.ravel().astype(np.int32)

    mask = rows != cols  # remove self-loops
    rows = rows[mask]
    cols = cols[mask]

    return np.column_stack([rows, cols])  # (K, 2) int32


# ---------------------------------------------------------------------------
# Bidirectionality filter (merge step)
# ---------------------------------------------------------------------------

def filter_mutual_edges(
    directed: np.ndarray,
    N: int,
    labels: np.ndarray | None,
) -> list[list[int]]:
    """Keep only mutual edges and build the final adjacency list.

    Parameters
    ----------
    directed:
        ``(E, 2)`` int32 array of all directed candidate edges.
    N:
        Total number of nodes.
    labels:
        Optional ``(N,)`` int array.  Same-class edges are removed when given.

    Returns
    -------
    list[list[int]]
        ``adj[i]`` contains the neighbours of node ``i`` (sorted, deduplicated).
    """
    logger.info("Total directed edges before filter: %d", len(directed))

    rows = directed[:, 0].astype(np.int64)
    cols = directed[:, 1].astype(np.int64)

    # Encode forward and reverse edges as int64 scalars
    encoded_fwd = rows * N + cols      # (E,)
    encoded_rev = cols * N + rows      # (E,) — reverse direction

    # Sort forward encodings once for O(log E) per lookup
    with Timer("Sorting forward edges", logger):
        fwd_sorted = np.sort(encoded_fwd)

    # Vectorised searchsorted check: is encoded_rev[k] in fwd_sorted?
    with Timer("Bidirectionality searchsorted", logger):
        pos = np.searchsorted(fwd_sorted, encoded_rev)
        in_range = pos < len(fwd_sorted)
        mutual = in_range & (fwd_sorted[np.minimum(pos, len(fwd_sorted) - 1)] == encoded_rev)

    logger.info("Mutual edges (directed count): %d", int(mutual.sum()))

    if labels is not None:
        same_class = labels[directed[:, 0]] == labels[directed[:, 1]]
        mutual &= ~same_class
        logger.info("After label filter: %d mutual edges", int(mutual.sum()))

    mutual_rows = directed[mutual, 0]
    mutual_cols = directed[mutual, 1]

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(N)]
    for i, j in tqdm(zip(mutual_rows.tolist(), mutual_cols.tolist()),
                     total=int(mutual.sum()), desc="Building adj list"):
        adj[i].append(j)

    # Sort and deduplicate
    for i in range(N):
        if adj[i]:
            adj[i] = sorted(set(adj[i]))

    return adj


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_graph_stats(adj: list[list[int]], N: int) -> dict:
    degrees = [len(a) for a in adj]
    total = sum(degrees)
    return {
        "N": N,
        "total_directed_edges": total,
        "approx_undirected_edges": total // 2,
        "avg_degree": round(total / N, 3) if N else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "min_degree": int(min(degrees)) if degrees else 0,
        "isolated_nodes": int(sum(1 for d in degrees if d == 0)),
    }


# ---------------------------------------------------------------------------
# METIS text file writer
# ---------------------------------------------------------------------------

def write_metis_file(adj: list[list[int]], N: int, path: str) -> None:
    edge_set: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            edge_set.add((min(i, j), max(i, j)))
    with open(path, "w") as fh:
        fh.write(f"{N} {len(edge_set)}\n")
        for nbrs in adj:
            fh.write(" ".join(str(j + 1) for j in nbrs) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    # ------------------------------------------------------------------
    # Merge mode: collect all shard directed-edge arrays, filter, save
    # ------------------------------------------------------------------
    if args.merge:
        logger.info("Merge mode: assembling %d shards…", args.num_shards)
        # We need N from somewhere — peek at the neighbors file header
        neighbors = np.load(args.neighbors_path, mmap_mode="r")
        N = neighbors.shape[0]
        del neighbors  # free; we only needed N

        shard_arrays = []
        for sid in range(args.num_shards):
            p = os.path.join(args.output_dir, f"directed_edges_shard{sid}_of{args.num_shards}.npy")
            shard_arrays.append(np.load(p))
            logger.info("  loaded shard %d: %d edges", sid, len(shard_arrays[-1]))

        directed = np.concatenate(shard_arrays, axis=0)
        logger.info("Total directed edges: %d", len(directed))
        del shard_arrays

        labels = _load_labels(args)
        with Timer("Filtering mutual edges", logger):
            adj = filter_mutual_edges(directed, N, labels)

        _save_outputs(adj, N, args)

        # Clean up shard files
        for sid in range(args.num_shards):
            os.remove(os.path.join(args.output_dir, f"directed_edges_shard{sid}_of{args.num_shards}.npy"))
        logger.info("Shard edge files removed.")
        return

    # ------------------------------------------------------------------
    # Normal / per-shard mode
    # ------------------------------------------------------------------
    with Timer("Loading neighbors.npy", logger):
        neighbors = np.load(args.neighbors_path, mmap_mode="r")
    N, top_k = neighbors.shape
    logger.info("Neighbors: shape=%s  dtype=%s", neighbors.shape, neighbors.dtype)

    if args.p + args.m > top_k:
        raise ValueError(
            f"p + m = {args.p} + {args.m} = {args.p + args.m} > top_k={top_k}. "
            "Increase --top_k in Stage 2 or reduce --m."
        )

    rows_per_shard = (N + args.num_shards - 1) // args.num_shards
    shard_start = args.shard_id * rows_per_shard
    shard_end = min(shard_start + rows_per_shard, N)
    logger.info("Shard %d/%d → rows [%d, %d)", args.shard_id, args.num_shards, shard_start, shard_end)

    with Timer("Extracting directed edges", logger):
        directed = extract_directed_edges(neighbors, args.p, args.m, shard_start, shard_end)
    logger.info("Directed edges for this shard: %d", len(directed))

    if args.num_shards > 1:
        # Write partial directed edges; merge step will do bidirectionality
        shard_path = os.path.join(
            args.output_dir, f"directed_edges_shard{args.shard_id}_of{args.num_shards}.npy"
        )
        np.save(shard_path, directed)
        logger.info("Shard edge file saved → %s", shard_path)
        return

    # Single-shard path: do full pipeline in one shot
    labels = _load_labels(args)
    with Timer("Filtering mutual edges", logger):
        adj = filter_mutual_edges(directed, N, labels)

    _save_outputs(adj, N, args)


def _load_labels(args: argparse.Namespace) -> np.ndarray | None:
    if not args.use_labels:
        return None
    if args.labels_path is None:
        raise ValueError("--use_labels requires --labels_path")
    labels = np.load(args.labels_path).astype(np.int32)
    logger.info("Labels shape: %s  unique: %d", labels.shape, len(np.unique(labels)))
    return labels


def _save_outputs(adj: list[list[int]], N: int, args: argparse.Namespace) -> None:
    stats = compute_graph_stats(adj, N)
    logger.info("Graph stats: %s", json.dumps(stats))

    adj_path = os.path.join(args.output_dir, "adjacency_list.pkl")
    with open(adj_path, "wb") as fh:
        pickle.dump(adj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Adjacency list → %s", adj_path)

    with open(os.path.join(args.output_dir, "graph_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    if args.write_metis_format:
        metis_path = os.path.join(args.output_dir, "graph.metis")
        write_metis_file(adj, N, metis_path)
        logger.info("METIS graph file → %s", metis_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: Build sparse adjacency graph from rank matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None)
    p.add_argument("--neighbors_path", default="outputs/imagenet1k_b3/ranks/neighbors.npy")
    p.add_argument("--output_dir", default="outputs/imagenet1k_b3/graph")
    p.add_argument("--p", type=int, default=30, help="Skip top-p neighbors.")
    p.add_argument("--m", type=int, default=100, help="Window size [p, p+m).")
    p.add_argument("--use_labels", action="store_true")
    p.add_argument("--labels_path", default=None)
    p.add_argument("--write_metis_format", action="store_true")
    # Sharding
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total SLURM array tasks for Stage 3.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Zero-based shard index.")
    p.add_argument("--merge", action="store_true",
                   help="Merge all shard edge arrays into adjacency_list.pkl.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        from utils import load_yaml_config
        cfg = load_yaml_config(args.config)
        g = cfg.get("graph", {})
        if args.p == 30:
            args.p = g.get("p", args.p)
        if args.m == 100:
            args.m = g.get("m", args.m)
        if not args.use_labels:
            args.use_labels = g.get("use_labels", False)
        if args.labels_path is None:
            args.labels_path = g.get("labels_path")
    logger.info("Configuration: %s", vars(args))
    main(args)
