"""
Feature Extraction Script — Pre-extract frozen ViT features to disk.

Loads a pre-trained SSL checkpoint, reconstructs the ViT backbone (no projector),
and saves [CLS] token and mean(patch_tokens) per image for each split.
These can then be loaded by probe_trainer.py for fast probe training.

Usage:
    python eval/extract_features.py \
        --checkpoint_path data/checkpoints/.../checkpoint_lastLeJEPA_e99_inet100_6LV.pth \
        --model_name vit_base_patch16_224.dino \
        --output_dir data/features/LeJEPA_inet100 \
        --dataset inet100 \
        --splits train val \
        --batch_size 256 \
        --num_workers 4
"""

import argparse
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import timm
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_backbone(checkpoint_path: str, model_name: str, device: str = "cuda") -> nn.Module:
    """
    Load a ViT backbone from a legacy .pth checkpoint.

    Handles torch.compile key prefixes (_orig_mod.) automatically.
    Returns a clean timm model ready for inference (no compile, no grad checkpointing).
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True if model_name.startswith("vit") else False,
    )

    # Load backbone_only weights (preferred) or fall back to extracting from full encoder
    if "backbone_only" in state:
        raw_sd = state["backbone_only"]
    elif "encoder" in state:
        raw_sd = {
            k.replace("backbone.", "", 1): v
            for k, v in state["encoder"].items()
            if k.startswith("backbone.")
        }
    elif "state_dict" in state:
        # Lightning checkpoint format
        raw_sd = {
            k.replace("encoder.backbone.", "", 1): v
            for k, v in state["state_dict"].items()
            if "encoder.backbone." in k
        }
    else:
        raise ValueError(
            f"Checkpoint has unrecognized keys: {list(state.keys())[:10]}..."
        )

    # Strip _orig_mod. prefix that torch.compile adds
    clean_sd = {}
    for k, v in raw_sd.items():
        clean_key = k.replace("_orig_mod.", "")
        clean_sd[clean_key] = v

    info = backbone.load_state_dict(clean_sd, strict=False)
    if info.missing_keys:
        logger.warning(f"Missing keys when loading backbone: {info.missing_keys}")
    if info.unexpected_keys:
        logger.warning(f"Unexpected keys when loading backbone: {info.unexpected_keys}")

    backbone.eval()
    backbone.requires_grad_(False)
    backbone = backbone.to(device)

    logger.info(
        f"Loaded backbone from {checkpoint_path} "
        f"(feat_dim={backbone.num_features}, model={model_name})"
    )
    return backbone


def build_dataset(dataset_name: str, split: str):
    """
    Build a dataset for feature extraction.

    Uses the existing HFDataset in eval-mode (V_global=1, V_local=0)
    which applies Resize(256) -> CenterCrop(224) -> Normalize.

    Override this function to add new dataset sources.
    """
    # Import here to keep this script runnable standalone
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ds import HFDataset

    ds = HFDataset(
        split=split,
        V_global=1,
        V_local=0,
        global_img_size=224,
        local_img_size=96,
        dataset=dataset_name,
    )
    return ds


def collate_single_view(batch):
    """Collate for single-view datasets. Each item is ([img_tensor], label)."""
    imgs = torch.stack([item[0][0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return imgs, labels


@torch.no_grad()
def extract_split(
    backbone: nn.Module,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict:
    """
    Run frozen backbone over one dataset split.

    Returns dict with:
        cls_features: (N, D) — [CLS] token
        patch_features: (N, D) — mean of patch tokens
        labels: (N,)
    """
    ds = build_dataset(dataset_name, split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_single_view,
    )

    all_cls = []
    all_patch = []
    all_labels = []

    logger.info(f"Extracting features for split='{split}' ({len(ds)} samples)")

    for imgs, labels in tqdm.tqdm(loader, desc=f"Extracting {split}"):
        imgs = imgs.to(device, non_blocking=True)

        with autocast(device, dtype=torch.bfloat16):
            # forward_features returns (B, num_tokens, D) for ViTs
            # tokens = [CLS, patch_1, patch_2, ..., patch_196]
            tokens = backbone.forward_features(imgs)

        # Extract [CLS] and mean(patches) in float32 for stable storage
        cls_token = tokens[:, 0, :].float().cpu()
        patch_mean = tokens[:, 1:, :].float().mean(dim=1).cpu()

        all_cls.append(cls_token)
        all_patch.append(patch_mean)
        all_labels.append(labels)

    return {
        "cls_features": torch.cat(all_cls, dim=0),
        "patch_features": torch.cat(all_patch, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract frozen ViT features to disk")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to .pth or .ckpt checkpoint file",
    )
    parser.add_argument(
        "--model_name", type=str, default="vit_base_patch16_224.dino",
        help="timm model name (must match architecture used during SSL training)",
    )
    parser.add_argument(
        "--dataset", type=str, default="inet100",
        help="Dataset name passed to HFDataset (e.g. inet100, cifar10)",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val"],
        help="Dataset splits to extract (e.g. train val test)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save extracted feature .pt files",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    os.makedirs(args.output_dir, exist_ok=True)

    backbone = load_backbone(args.checkpoint_path, args.model_name, args.device)

    # Verify no gradient flow
    trainable = sum(p.requires_grad for p in backbone.parameters())
    assert trainable == 0, f"Backbone has {trainable} trainable params — should be 0"

    for split in args.splits:
        features = extract_split(
            backbone=backbone,
            dataset_name=args.dataset,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )

        # Save to disk
        for key, tensor in features.items():
            path = os.path.join(args.output_dir, f"{split}_{key}.pt")
            torch.save(tensor, path)
            logger.info(f"Saved {path} — shape {tensor.shape}, dtype {tensor.dtype}")

    logger.info(f"Feature extraction complete. Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
