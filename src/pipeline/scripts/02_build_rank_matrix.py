"""Stage 2 — Build a sparse top-k rank/neighbor matrix using FAISS.

Reads ``embeddings.npy`` produced by Stage 1 and writes:

  <output_dir>/neighbors.npy  — int32, shape (N, top_k): indices of the
                                 top-k most similar samples for every query

Memory design
-------------
* Embeddings are L2-normalised once and the full normalised matrix is kept in
  RAM (1.28M × 1024 × 4 bytes ≈ 5 GB for ViT-L).  If RAM is tight, reduce
  ``--chunk_size`` or use ``--num_shards`` to process only a slice.
* ``neighbors.npy`` is written via a memory-mapped array so it never needs to
  be fully resident.

SLURM sharding
--------------
Run multiple jobs in parallel with ``--num_shards N --shard_id i`` (0-indexed).
Each job writes a shard file ``neighbors_shard{i}_of{N}.npy``; stitch them
afterwards with the provided ``--stitch`` flag on the driver node.

Usage
-----
    # Single-node (all samples)
    python 02_build_rank_matrix.py \\
        --embeddings_path outputs/imagenet1k_b3/embeddings/embeddings.npy \\
        --output_dir      outputs/imagenet1k_b3/ranks \\
        --top_k 130 --use_gpu_faiss

    # Sharded (4 SLURM jobs)
    python 02_build_rank_matrix.py ... --num_shards 4 --shard_id 0
    python 02_build_rank_matrix.py ... --num_shards 4 --shard_id 1
    ...
    python 02_build_rank_matrix.py ... --num_shards 4 --stitch
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import Timer, build_faiss_index, ensure_dir, load_embeddings, normalize_embeddings, setup_logging

logger = setup_logging("02_build_rank_matrix")


# ---------------------------------------------------------------------------
# Core search routine
# ---------------------------------------------------------------------------

def build_neighbor_matrix(
    emb_norm: np.ndarray,
    top_k: int,
    chunk_size: int,
    use_gpu: bool,
    gpu_id: int,
    shard_start: int,
    shard_end: int,
    out_array: np.ndarray,
    score_array: np.ndarray,
) -> None:
    """Fill *out_array* and *score_array* with top-k neighbor indices and scores.

    Parameters
    ----------
    emb_norm:
        Full L2-normalised embedding matrix, shape ``(N, D)``, float32.
    top_k:
        How many nearest neighbors to retrieve per query.
    chunk_size:
        Number of queries sent to ``index.search`` per call.
    use_gpu:
        Whether to push the FAISS index to a GPU.
    gpu_id:
        Which GPU device.
    shard_start / shard_end:
        Row slice of *emb_norm* that this call is responsible for.
    out_array:
        Pre-allocated int32 array of shape ``(shard_end - shard_start, top_k)``
        that will receive the neighbor indices.
    score_array:
        Pre-allocated float32 array of shape ``(shard_end - shard_start, top_k)``
        that will receive the cosine similarity scores (inner products after L2 norm).
    """
    with Timer("Building FAISS index", logger):
        index = build_faiss_index(emb_norm, use_gpu=use_gpu, gpu_id=gpu_id)

    shard_size = shard_end - shard_start
    logger.info(
        "Searching top-%d neighbors for rows [%d, %d) (%d queries) in chunks of %d",
        top_k, shard_start, shard_end, shard_size, chunk_size,
    )

    n_chunks = (shard_size + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        q_start = shard_start + chunk_idx * chunk_size
        q_end = min(q_start + chunk_size, shard_end)
        query = emb_norm[q_start:q_end]  # (B, D)

        # index.search returns (distances, indices) each (B, top_k)
        scores, I = index.search(query, top_k)

        local_start = q_start - shard_start
        local_end = q_end - shard_start
        out_array[local_start:local_end] = I.astype(np.int32)
        score_array[local_start:local_end] = scores.astype(np.float32)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == 0:
            logger.info(
                "  chunk %d/%d  (rows %d–%d)",
                chunk_idx + 1, n_chunks, q_start, q_end - 1,
            )


# ---------------------------------------------------------------------------
# Stitch shards back into a single file
# ---------------------------------------------------------------------------

def stitch_shards(output_dir: str, num_shards: int, top_k: int, N: int) -> None:
    """Concatenate per-shard neighbor and score files into single output files."""
    out_path = os.path.join(output_dir, "neighbors.npy")
    score_out_path = os.path.join(output_dir, "neighbor_scores.npy")
    logger.info("Stitching %d shards → %s and %s", num_shards, out_path, score_out_path)

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.int32, shape=(N, top_k)
    )
    score_out = np.lib.format.open_memmap(
        score_out_path, mode="w+", dtype=np.float32, shape=(N, top_k)
    )

    written = 0
    for sid in range(num_shards):
        shard_path = os.path.join(output_dir, f"neighbors_shard{sid}_of{num_shards}.npy")
        score_shard_path = os.path.join(output_dir, f"neighbors_scores_shard{sid}_of{num_shards}.npy")
        shard = np.load(shard_path, mmap_mode="r")
        score_shard = np.load(score_shard_path, mmap_mode="r")
        n = shard.shape[0]
        out[written: written + n] = shard
        score_out[written: written + n] = score_shard
        written += n
        logger.info("  shard %d/%d: %d rows", sid + 1, num_shards, n)

    out.flush()
    score_out.flush()
    logger.info("Stitched %d rows → %s", written, out_path)
    logger.info("Stitched %d rows → %s", written, score_out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    # ------------------------------------------------------------------
    # Stitch-only mode
    # ------------------------------------------------------------------
    if args.stitch:
        emb = load_embeddings(args.embeddings_path)
        N = emb.shape[0]
        stitch_shards(args.output_dir, args.num_shards, args.top_k, N)
        return

    # ------------------------------------------------------------------
    # Load & normalise embeddings
    # ------------------------------------------------------------------
    with Timer("Loading embeddings", logger):
        emb_raw = load_embeddings(args.embeddings_path, mmap_mode="r")
        N, D = emb_raw.shape
        logger.info("Raw embeddings: shape=%s dtype=%s", emb_raw.shape, emb_raw.dtype)

    with Timer("Normalising embeddings", logger):
        emb_norm = normalize_embeddings(emb_raw, chunk_size=65536)
    logger.info("Normalised embeddings: shape=%s dtype=%s", emb_norm.shape, emb_norm.dtype)

    # ------------------------------------------------------------------
    # Determine row slice for this shard
    # ------------------------------------------------------------------
    rows_per_shard = (N + args.num_shards - 1) // args.num_shards
    shard_start = args.shard_id * rows_per_shard
    shard_end = min(shard_start + rows_per_shard, N)
    shard_size = shard_end - shard_start
    logger.info("Shard %d/%d → rows [%d, %d)  (%d queries)",
                args.shard_id, args.num_shards, shard_start, shard_end, shard_size)

    # ------------------------------------------------------------------
    # Allocate output
    # ------------------------------------------------------------------
    if args.num_shards == 1:
        out_path = os.path.join(args.output_dir, "neighbors.npy")
        score_path = os.path.join(args.output_dir, "neighbor_scores.npy")

    else:
        out_path = os.path.join(
            args.output_dir, f"neighbors_shard{args.shard_id}_of{args.num_shards}.npy"
        )
        score_path = os.path.join(
            args.output_dir, f"neighbors_scores_shard{args.shard_id}_of{args.num_shards}.npy"
        )

    logger.info("Output: %s  shape=(%d, %d) int32", out_path, shard_size, args.top_k)
    out_array = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.int32, shape=(shard_size, args.top_k)
    )
    score_array = np.lib.format.open_memmap(
        score_path, mode="w+", dtype=np.float32, shape=(shard_size, args.top_k)
    )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    with Timer("FAISS neighbor search", logger):
        build_neighbor_matrix(
            emb_norm=emb_norm,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            use_gpu=args.use_gpu_faiss,
            gpu_id=args.gpu_id,
            shard_start=shard_start,
            shard_end=shard_end,
            out_array=out_array,
            score_array=score_array,
        )

    out_array.flush()
    score_array.flush()
    logger.info("Done. Neighbor matrix written to %s", out_path)

    # Quick sanity check
    check = np.load(out_path, mmap_mode="r")
    logger.info(
        "Sanity check — shape: %s  min: %d  max: %d  dtype: %s",
        check.shape, int(check.min()), int(check.max()), check.dtype,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: Build sparse top-k neighbor matrix via FAISS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Optional path to default.yaml.")
    p.add_argument(
        "--embeddings_path",
        default="outputs/imagenet1k_b3/embeddings/embeddings.npy",
        help="Path to embeddings.npy from Stage 1.",
    )
    p.add_argument("--output_dir", default="outputs/imagenet1k_b3/ranks",
                   help="Directory for neighbors.npy output.")
    p.add_argument("--top_k", type=int, default=130,
                   help="Number of nearest neighbors to retrieve per query.")
    p.add_argument("--chunk_size", type=int, default=32768,
                   help="Number of queries per FAISS search call.")
    p.add_argument("--use_gpu_faiss", action="store_true",
                   help="Move FAISS index to GPU.")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="GPU device index for FAISS (when --use_gpu_faiss).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards (for parallel SLURM execution).")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Zero-based index of the shard to process.")
    p.add_argument("--stitch", action="store_true",
                   help="Stitch per-shard files into a single neighbors.npy (run after all shards finish).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        from utils import load_yaml_config
        cfg = load_yaml_config(args.config)
        r = cfg.get("ranking", {})
        if args.top_k == 130:
            args.top_k = r.get("top_k", args.top_k)
        if args.chunk_size == 32768:
            args.chunk_size = r.get("chunk_size", args.chunk_size)
        if not args.use_gpu_faiss:
            args.use_gpu_faiss = r.get("use_gpu_faiss", args.use_gpu_faiss)

    logger.info("Configuration: %s", vars(args))
    main(args)
