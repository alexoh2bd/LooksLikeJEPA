"""Stage 1 — Extract teacher embeddings from ImageNet-1K.

Runs a pretrained timm ViT over the full ImageNet-1K training split stored as
HuggingFace parquet shards and saves:

  <output_dir>/embeddings.npy       — float32, shape (N, D), memory-mapped
  <output_dir>/metadata.json        — list of {index, label} per sample

Parallel sharding
-----------------
Pass ``--num_shards N --shard_id i`` (0-indexed) to split the 294 parquet
files across N SLURM array tasks that run simultaneously.  Each task writes:

  <output_dir>/embeddings_shard{i}_of{N}.npy
  <output_dir>/metadata_shard{i}_of{N}.json

After all array tasks finish, run the stitch step on the driver node::

    python 01_extract_embeddings.py ... --stitch --num_shards N

This concatenates the shard files into the final ``embeddings.npy`` and
``metadata.json``, then deletes the temporary shard files.

Usage
-----
    # Single job (no sharding)
    python 01_extract_embeddings.py \\
        --data_dir  data/hub/.../data \\
        --output_dir outputs/imagenet1k_b3/embeddings

    # SLURM array (4 shards)
    python 01_extract_embeddings.py ... --num_shards 4 --shard_id $SLURM_ARRAY_TASK_ID

    # Stitch after array completes
    python 01_extract_embeddings.py ... --stitch --num_shards 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(__file__))
from utils import Timer, ensure_dir, set_seed, setup_logging

logger = setup_logging("01_extract_embeddings")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFImageNetDataset(torch.utils.data.Dataset):
    """HuggingFace parquet-backed ImageNet-1K dataset for inference.

    Parameters
    ----------
    parquet_files:
        Explicit list of ``.parquet`` file paths to load (allows sharding by
        passing a subset of the 294 shard files).
    transform:
        torchvision transform applied to each PIL image.
    """

    def __init__(self, parquet_files: list[str], transform: nn.Module | None = None) -> None:
        from datasets import load_dataset

        if not parquet_files:
            raise FileNotFoundError("No parquet files provided.")
        logger.info("Loading %d parquet shard(s)…", len(parquet_files))

        with Timer("Loading HuggingFace dataset", logger):
            self.ds = load_dataset(
                "parquet",
                data_files={"train": parquet_files},
                split="train",
            )

        self.transform = transform
        logger.info("Dataset size: %d samples", len(self.ds))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        label: int = row["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _split_parquet_files(data_dir: str, num_shards: int, shard_id: int) -> list[str]:
    """Return the subset of parquet files assigned to *shard_id*."""
    pattern = os.path.join(data_dir, "train-*.parquet")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No train parquet shards found in: {data_dir}")
    # Distribute files round-robin across shards for balanced sizes
    return [f for i, f in enumerate(all_files) if i % num_shards == shard_id]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen3_vl_model(
    model_path: str,
    device: torch.device,
    embedding_dim: int | None = None,
) -> tuple[nn.Module, int]:
    """Load Qwen3-VL-Embedding-2B as the teacher encoder.

    Uses the Qwen3VLEmbedder from the model repo. Expects PIL images.
    Returns ``(embedder, actual_embedding_dim)``.
    """
    from huggingface_hub import snapshot_download

    # Compatibility: inject check_model_inputs if missing (older transformers)
    import transformers.utils.generic as _tg
    if not hasattr(_tg, "check_model_inputs"):

        def _check_model_inputs(f):
            return f

        _tg.check_model_inputs = _check_model_inputs

    logger.info("Loading Qwen3-VL-Embedding: %s", model_path)
    repo_path = snapshot_download(model_path)
    sys.path.insert(0, repo_path)
    try:
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    finally:
        sys.path.pop(0)

    embedder = Qwen3VLEmbedder(model_name_or_path=model_path)
    embedder.model.to(device)
    embedder.model.eval()

    # Infer embedding dim from a dummy forward (image-only)
    with torch.no_grad():
        dummy_inputs = [{"image": __import__("PIL.Image").Image.new("RGB", (224, 224), color=0)}]
        emb = embedder.process(dummy_inputs, normalize=False)
        actual_dim = emb.shape[-1]

    if embedding_dim is not None and actual_dim != embedding_dim:
        logger.warning("Expected dim=%d, got dim=%d. Using %d.", embedding_dim, actual_dim, actual_dim)
    logger.info("Qwen3-VL teacher ready. Embedding dim: %d", actual_dim)
    return embedder, actual_dim


def load_teacher_model(
    model_name: str,
    checkpoint_path: str | None,
    embedding_dim: int | None,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """Load a timm ViT backbone as the teacher encoder.

    Supports Lightning ``.ckpt`` checkpoints by stripping common key prefixes.
    Returns ``(model, actual_embedding_dim)``.
    """
    import timm

    logger.info("Loading timm model: %s", model_name)
    model = timm.create_model(
        model_name,
        pretrained=(checkpoint_path is None),
        num_classes=0,
    )

    if checkpoint_path is not None:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        def _strip(key: str) -> str:
            for prefix in ("encoder.backbone.", "backbone.", "teacher.", "model."):
                if key.startswith(prefix):
                    return key[len(prefix):]
            return key

        state = {_strip(k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s…", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s…", len(unexpected), unexpected[:5])

    model.eval().to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        actual_dim = model(dummy).shape[-1]

    if embedding_dim is not None and actual_dim != embedding_dim:
        logger.warning("Expected dim=%d, got dim=%d. Using %d.", embedding_dim, actual_dim, actual_dim)
    logger.info("Teacher ready. Embedding dim: %d", actual_dim)
    return model, actual_dim


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_inference(
    args: argparse.Namespace,
    parquet_files: list[str],
    out_emb_path: str,
    out_meta_path: str,
) -> None:
    """Run inference on *parquet_files* and write results to *out_emb_path*."""
    device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
    primary = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() and device_ids else "cpu")
    logger.info("Primary device: %s", primary)

    use_qwen = args.teacher_type == "qwen3_vl"

    if use_qwen:
        # Qwen3-VL expects PIL images; its processor handles preprocessing
        transform = None
    else:
        transform = v2.Compose([
            v2.Resize(args.img_size + 32, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(args.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    dataset = HFImageNetDataset(parquet_files, transform=transform)
    N = len(dataset)

    def collate_pil_batch(batch: list) -> tuple:
        """Collate (PIL.Image, label) pairs into (list[PIL.Image], LongTensor)."""
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, labels

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(primary.type == "cuda" and not use_qwen),
        drop_last=False,
        collate_fn=collate_pil_batch if use_qwen else None,
    )

    if use_qwen:
        model, emb_dim = load_qwen3_vl_model(
            args.qwen_model_path, primary, args.embedding_dim
        )
    else:
        model, emb_dim = load_teacher_model(
            args.model_name, args.checkpoint_path, args.embedding_dim, primary
        )
        if torch.cuda.is_available() and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            logger.info("DataParallel on GPUs: %s", device_ids)

    logger.info("Allocating memmap %s  shape=(%d, %d)", out_emb_path, N, emb_dim)
    embeddings = np.lib.format.open_memmap(out_emb_path, mode="w+", dtype=np.float32, shape=(N, emb_dim))

    all_labels: list[int] = []
    written = 0

    with Timer("Inference", logger):
        if use_qwen:
            with torch.no_grad():
                for batch_idx, (imgs, labels) in enumerate(loader):
                    # imgs is list of PIL when transform=None
                    inputs = [{"image": img, "instruction":"Represent the given image for classification"} for img in imgs]
                    feats = model.process(inputs, normalize=False).cpu().float().numpy()
                    B = feats.shape[0]
                    embeddings[written: written + B] = feats
                    all_labels.extend(labels.tolist())
                    written += B
                    if (batch_idx + 1) % 100 == 0:
                        logger.info("  [%d/%d] wrote %d/%d samples", batch_idx + 1, len(loader), written, N)
        else:
            with torch.no_grad():
                for batch_idx, (imgs, labels) in enumerate(loader):
                    imgs = imgs.to(primary, non_blocking=True)
                    feats = model(imgs).cpu().float().numpy()
                    B = feats.shape[0]
                    embeddings[written: written + B] = feats
                    all_labels.extend(labels.tolist())
                    written += B
                    if (batch_idx + 1) % 100 == 0:
                        logger.info("  [%d/%d] wrote %d/%d samples", batch_idx + 1, len(loader), written, N)

    embeddings.flush()
    del embeddings
    logger.info("Wrote %d embeddings → %s", written, out_emb_path)

    with open(out_meta_path, "w") as fh:
        json.dump([{"index": i, "label": lbl} for i, lbl in enumerate(all_labels)], fh)
    logger.info("Metadata → %s", out_meta_path)


# ---------------------------------------------------------------------------
# Stitch mode
# ---------------------------------------------------------------------------

def stitch_shards(output_dir: str, num_shards: int) -> None:
    """Concatenate per-shard embedding and metadata files."""
    logger.info("Stitching %d shards in %s", num_shards, output_dir)

    # Peek at shard 0 to get embedding dim
    shard0_emb = os.path.join(output_dir, f"embeddings_shard0_of{num_shards}.npy")
    s0 = np.load(shard0_emb, mmap_mode="r")
    emb_dim = s0.shape[1]

    # Count total rows
    total_rows = 0
    for sid in range(num_shards):
        p = os.path.join(output_dir, f"embeddings_shard{sid}_of{num_shards}.npy")
        total_rows += np.load(p, mmap_mode="r").shape[0]
    logger.info("Total rows across shards: %d  dim=%d", total_rows, emb_dim)

    out_emb = np.lib.format.open_memmap(
        os.path.join(output_dir, "embeddings.npy"), mode="w+",
        dtype=np.float32, shape=(total_rows, emb_dim),
    )
    all_meta: list[dict] = []
    written = 0

    for sid in range(num_shards):
        emb_path = os.path.join(output_dir, f"embeddings_shard{sid}_of{num_shards}.npy")
        meta_path = os.path.join(output_dir, f"metadata_shard{sid}_of{num_shards}.json")
        shard_emb = np.load(emb_path, mmap_mode="r")
        with open(meta_path) as fh:
            shard_meta = json.load(fh)

        n = shard_emb.shape[0]
        out_emb[written: written + n] = shard_emb
        # Re-index metadata to global indices
        for i, entry in enumerate(shard_meta):
            all_meta.append({"index": written + i, "label": entry["label"]})
        written += n
        logger.info("  shard %d/%d: %d rows (running total %d)", sid, num_shards, n, written)

    out_emb.flush()
    logger.info("Stitched embeddings → %s", os.path.join(output_dir, "embeddings.npy"))

    meta_out = os.path.join(output_dir, "metadata.json")
    with open(meta_out, "w") as fh:
        json.dump(all_meta, fh)
    logger.info("Stitched metadata → %s", meta_out)

    # Clean up shard files
    for sid in range(num_shards):
        os.remove(os.path.join(output_dir, f"embeddings_shard{sid}_of{num_shards}.npy"))
        os.remove(os.path.join(output_dir, f"metadata_shard{sid}_of{num_shards}.json"))
    logger.info("Shard files removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.config:
        from utils import load_yaml_config
        cfg = load_yaml_config(args.config)
        t = cfg.get("teacher", {})
        d = cfg.get("data", {})
        if args.model_name == "vit_large_patch16_224":
            args.model_name = t.get("model_name", args.model_name)
        args.teacher_type = t.get("teacher_type", args.teacher_type)
        args.qwen_model_path = t.get("qwen_model_path", args.qwen_model_path)
        if args.checkpoint_path is None:
            args.checkpoint_path = t.get("checkpoint_path")
        if args.embedding_dim is None:
            args.embedding_dim = t.get("embedding_dim")
        if "data/hub" in args.data_dir:
            args.data_dir = d.get("data_dir", args.data_dir)
        args.num_workers = d.get("num_workers", args.num_workers)
        if args.batch_size == 256:
            args.batch_size = d.get("batch_size_inference", args.batch_size)
    # Infer teacher_type from model_name for backward compatibility
    if args.model_name == "qwen3_vl":
        args.teacher_type = "qwen3_vl"

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    logger.info("Configuration: %s", vars(args))

    # Stitch-only mode
    if args.stitch:
        stitch_shards(args.output_dir, args.num_shards)
        return

    # Determine which parquet files to process
    parquet_files = _split_parquet_files(args.data_dir, args.num_shards, args.shard_id)
    logger.info(
        "Shard %d/%d → %d parquet files",
        args.shard_id, args.num_shards, len(parquet_files),
    )

    if args.num_shards == 1:
        out_emb = os.path.join(args.output_dir, "embeddings.npy")
        out_meta = os.path.join(args.output_dir, "metadata.json")
    else:
        out_emb = os.path.join(args.output_dir, f"embeddings_shard{args.shard_id}_of{args.num_shards}.npy")
        out_meta = os.path.join(args.output_dir, f"metadata_shard{args.shard_id}_of{args.num_shards}.json")

    run_inference(args, parquet_files, out_emb, out_meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Extract teacher embeddings from ImageNet-1K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None)
    p.add_argument(
        "--teacher_type",
        choices=("timm", "qwen3_vl"),
        default="timm",
        help="Teacher encoder type: timm (ViT/OpenCLIP) or qwen3_vl.",
    )
    p.add_argument("--model_name", default="vit_large_patch16_224",
                   help="timm model name (ignored when teacher_type=qwen3_vl).")
    p.add_argument(
        "--qwen_model_path",
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="HuggingFace model path for Qwen3-VL (when teacher_type=qwen3_vl).",
    )
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--embedding_dim", type=int, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument(
        "--data_dir",
        default=(
            "data/hub/datasets--ILSVRC--imagenet-1k/snapshots/"
            "49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data"
        ),
    )
    p.add_argument("--output_dir", default="outputs/imagenet1k_b3/embeddings")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device_ids", default="0",
                   help="Comma-separated CUDA device IDs.")
    p.add_argument("--seed", type=int, default=42)
    # Sharding
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total SLURM array tasks for Stage 1.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Zero-based shard index (set to $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--stitch", action="store_true",
                   help="Stitch per-shard files into final embeddings.npy.")
    return p.parse_args()


if __name__ == "__main__":
    main()
