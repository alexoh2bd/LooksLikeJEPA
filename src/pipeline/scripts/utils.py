"""Shared utilities for the B3-LeJEPA batch mining pipeline."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent timestamp-prefixed format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that logs elapsed wall-clock time."""

    def __init__(self, description: str, logger: logging.Logger | None = None) -> None:
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.logger.info("Starting: %s", self.description)
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.logger.info("Finished: %s in %.2fs", self.description, self.elapsed)


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def load_embeddings(path: str, mmap_mode: str | None = "r") -> np.ndarray:
    """Load an embeddings array, optionally memory-mapped.

    Parameters
    ----------
    path:
        Path to a `.npy` file of shape ``(N, D)``.
    mmap_mode:
        Passed directly to ``np.load``. Use ``None`` to load fully into RAM.
    """
    return np.load(path, mmap_mode=mmap_mode)


def normalize_embeddings(emb: np.ndarray, chunk_size: int = 65536) -> np.ndarray:
    """Return a float32 L2-normalised copy of *emb*.

    Processes in chunks to avoid peak memory spikes for large arrays.
    The returned array is always a fresh, contiguous float32 array (not a
    memory-mapped view).

    Parameters
    ----------
    emb:
        Input array shape ``(N, D)``.
    chunk_size:
        Number of rows processed per iteration.
    """
    N, D = emb.shape
    out = np.empty((N, D), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = emb[start:end].astype(np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard against zero vectors
        out[start:end] = chunk / norms
    return out


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def build_faiss_index(
    emb_norm: np.ndarray,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> "faiss.Index":  # noqa: F821  (faiss imported lazily)
    """Build a ``faiss.IndexFlatIP`` from *pre-normalised* float32 embeddings.

    Because embeddings are L2-normalised, inner product == cosine similarity.

    Parameters
    ----------
    emb_norm:
        Float32 array of shape ``(N, D)`` with unit-norm rows.
    use_gpu:
        If True and a GPU is available, wrap the index with a GPU resource.
    gpu_id:
        Which GPU device to use when *use_gpu* is True.

    Returns
    -------
    faiss.Index
        A trained and populated index ready for ``.search()``.
    """
    import faiss  # deferred import so the rest of utils remains importable without faiss

    logger = setup_logging(__name__)
    D = emb_norm.shape[1]
    index = faiss.IndexFlatIP(D)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            logger.info("FAISS index moved to GPU %d", gpu_id)
        except Exception as exc:
            logger.warning("Failed to move FAISS to GPU (%s). Falling back to CPU.", exc)

    logger.info("Adding %d vectors (D=%d) to FAISS index…", emb_norm.shape[0], D)
    index.add(emb_norm)
    logger.info("FAISS index built with %d vectors.", index.ntotal)
    return index


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    try:
        import yaml  # PyYAML, available via hydra-core
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")

    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist. Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
