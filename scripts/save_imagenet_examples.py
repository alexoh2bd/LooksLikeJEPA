#!/usr/bin/env python3
"""Save 32 ImageNet-1K test images as example files for PCA viz and other scripts.

Usage:
    uv run python scripts/save_imagenet_examples.py
    uv run python scripts/save_imagenet_examples.py --num_images 64 --output_dir data/imagenet1k_examples
"""

import argparse
import os
import sys

# Add src for dataset path consistency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser(description="Save ImageNet-1K test images as examples")
    p.add_argument("--num_images", type=int, default=32)
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/imagenet1k_examples",
        help="Directory to save images",
    )
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    inet_dir = os.path.join(
        root,
        "data/hub/datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data",
    )
    import glob as glob_mod
    if args.split == "test":
        files = sorted(glob_mod.glob(inet_dir + "/test*.parquet"))
    else:
        files = sorted(glob_mod.glob(inet_dir + "/validation*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {args.split} parquet files in {inet_dir}")
    # Use first shard only for speed (~2.5k images)
    pattern = files[0]

    print(f"Loading {args.split} split from {pattern}")
    ds = load_dataset("parquet", data_files={args.split: pattern}, split=args.split)

    # Sample indices (stratified by label for diversity if possible)
    n = min(args.num_images, len(ds))
    rng = __import__("random").Random(args.seed)
    indices = rng.sample(range(len(ds)), n)

    os.makedirs(args.output_dir, exist_ok=True)
    img_key = "image" if "image" in ds.column_names else "img"

    for i, idx in enumerate(indices):
        row = ds[idx]
        img = row[img_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = row.get("label", -1)
        out_path = os.path.join(args.output_dir, f"imagenet_{i:03d}_label{label}.png")
        img.save(out_path)
        print(f"  Saved {out_path}")

    print(f"Saved {n} images to {args.output_dir}")


if __name__ == "__main__":
    main()
