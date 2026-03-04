"""
Few-Shot Linear Probe Transfer Evaluation on Frozen Image Embeddings.

Loads a pre-trained SSL checkpoint (ViT backbone + 3-layer MLP projector),
extracts and caches frozen features, then trains a linear classifier under
1% / 10% / 100% data regimes on six downstream datasets.

Two optimizer presets (from the paper appendix):
  L7  — Adam, lr=1e-2, wd=0, no schedule
  L9  — SGD(momentum=0.9), lr=1e-2, wd=1e-6, cosine schedule

Usage:
    python src/linear_probe.py \
        --checkpoint_path data/checkpoints/.../last.ckpt \
        --datasets cifar10 cifar100 dtd flowers102 food101 pets \
        --epochs 100 \
        --optim L7
"""

import gc
import argparse
import logging
from collections import defaultdict
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.ops import MLP
from torchvision.transforms import v2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "dtd": {
        "hf_path": "tanganke/dtd",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 47,
    },
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
        "image_key": "img",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 10,
    },
    "cifar100": {
        "hf_path": "uoft-cs/cifar100",
        "image_key": "img",
        "label_key": "fine_label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 100,
    },
    "flowers102": {
        "hf_path": "nelorth/oxford-flowers",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 102,
    },
    "food101": {
        "hf_path": "ethz/food101",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "validation",
        "num_classes": 101,
    },
    "pets": {
        "hf_path": "timm/oxford-iiit-pet",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 37,
    },
    "aircraft": {
        "hf_path": "randall-lab/fgvc-aircraft",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 100,
    },
    "cars": {
        "hf_path": "tanganke/stanford_cars",
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
        "num_classes": 196,
    }
}

# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` prefixes injected by ``torch.compile``."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_model(
    checkpoint_path: str,
    model_name: str = "vit_base_patch16_224.dino",
    proj_dim: int = 512,
    device: str = "cuda",
):
    """
    Load frozen backbone, projector, and output batch-norm from a checkpoint.

    Supports legacy ``.pth`` (keys: ``encoder`` or ``backbone_only``)
    and Lightning ``.ckpt`` (key: ``state_dict``).

    Returns (backbone, projector, output_bn, feat_dim) — all frozen, on *device*.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # _-- resolve raw state dicts for backbone / proj / output_bn ----
    if "backbone_only" in state:
        backbone_sd = _strip_compile_prefix(state["backbone_only"])
        # proj / output_bn live in "encoder"
        enc_sd = _strip_compile_prefix(state.get("encoder", {}))
        proj_sd = {
            k.replace("proj.", "", 1): v
            for k, v in enc_sd.items()
            if k.startswith("proj.")
        }
        bn_sd = {
            k.replace("output_bn.", "", 1): v
            for k, v in enc_sd.items()
            if k.startswith("output_bn.")
        }
    elif "encoder" in state:
        enc_sd = _strip_compile_prefix(state["encoder"])
        backbone_sd = {
            k.replace("backbone.", "", 1): v
            for k, v in enc_sd.items()
            if k.startswith("backbone.")
        }
    elif "state_dict" in state:
        full_sd = _strip_compile_prefix(state["state_dict"])
        backbone_sd = {
            k.replace("encoder.backbone.", "", 1): v
            for k, v in full_sd.items()
            if k.startswith("encoder.backbone.")
        }
    else:
        raise ValueError(
            f"Unrecognized checkpoint format — top-level keys: {list(state.keys())[:10]}"
        )

    # ---- build clean (non-compiled) modules ----
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=model_name.startswith("vit"),
    )
    feat_dim = backbone.num_features

    info = backbone.load_state_dict(backbone_sd, strict=False)
    if info.missing_keys:
        logger.warning("Backbone missing keys: %s", info.missing_keys)
    if info.unexpected_keys:
        logger.warning("Backbone unexpected keys: %s", info.unexpected_keys)

    # projector = MLP(
    #     in_channels=feat_dim,
    #     hidden_channels=[2048, 2048, proj_dim],
    #     norm_layer=nn.BatchNorm1d,
    # )
    # output_bn = nn.BatchNorm1d(proj_dim, affine=False)

    # if proj_sd:
    #     info = projector.load_state_dict(proj_sd, strict=False)
    #     if info.missing_keys:
    #         logger.warning("Projector missing keys: %s", info.missing_keys)
    # else:
    #     logger.warning("No projector weights found — projector will be random-init")

    # if bn_sd:
    #     info = output_bn.load_state_dict(bn_sd, strict=False)
    #     if info.missing_keys:
    #         logger.warning("Output BN missing keys: %s", info.missing_keys)

    backbone.eval()
    backbone.requires_grad_(False)
    backbone.to(device)

    logger.info(
        "Loaded model from %s  (backbone feat_dim=%d, proj_dim=%d)",
        checkpoint_path, feat_dim, proj_dim,
    )
    return backbone, feat_dim


# ---------------------------------------------------------------------------
# 3. Eval-mode image dataset (wraps a HuggingFace split)
# ---------------------------------------------------------------------------

EVAL_TRANSFORM = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    """Thin wrapper that applies *eval_transform* to a HuggingFace split."""

    def __init__(self, hf_dataset, image_key: str, label_key: str):
        self.ds = hf_dataset
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row[self.image_key].convert("RGB")
        img = EVAL_TRANSFORM(img)
        label = row[self.label_key]
        return img, label


def build_eval_dataset(dataset_name: str, split: str) -> ImageDataset:
    """Load a HuggingFace dataset split and wrap it with the eval transform."""
    cfg = DATASETS[dataset_name]

    if split == "train":
        split_str = cfg["train_split"]
    else:
        split_str = cfg["test_split"]

    hf_ds = load_dataset(cfg["hf_path"], split=split_str, trust_remote_code=True)
    logger.info(
        "Loaded %s split='%s' (%d samples)",
        dataset_name, split_str, len(hf_ds),
    )
    return ImageDataset(hf_ds, cfg["image_key"], cfg["label_key"])


# ---------------------------------------------------------------------------
# 4. Feature extraction + caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: ImageDataset,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Run the frozen encoder (and optionally projector) over *dataset*.

    Returns ``(features, labels)`` tensors on CPU in float32.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    all_feats, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  extracting"):
        imgs = imgs.to(device, non_blocking=True)

        with autocast(device, dtype=torch.bfloat16):
            emb = backbone(imgs)  # (B, feat_dim)


        all_feats.append(emb.float().cpu())
        all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logger.info("  cached %d features of dim %d", features.shape[0], features.shape[1])
    return features, labels


# ---------------------------------------------------------------------------
# 5. Stratified subsetting
# ---------------------------------------------------------------------------

def stratified_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
    seed: int = 0,
):
    """
    Return a stratified subset of (features, labels).

    For *fraction* == 1.0 returns the original tensors unchanged.
    """
    if fraction >= 1.0:
        return features, labels

    labels_np = labels.numpy()
    n_classes = len(np.unique(labels_np))
    # Convert fraction to absolute count, then clamp to n_classes minimum
    n_samples = max(int(len(labels_np) * fraction), n_classes)

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed,
    )
    idx, _ = next(sss.split(np.zeros(len(labels_np)), labels_np))
    idx = torch.from_numpy(idx)
    return features[idx], labels[idx]


# ---------------------------------------------------------------------------
# 6. ReLU + sparsity statistics
# ---------------------------------------------------------------------------

def apply_relu_and_report(
    train_feats: torch.Tensor,
    val_feats: torch.Tensor,
    dataset_name: str,
):
    """Apply post-hoc ReLU; print sparsity on val features before/after."""
    pre_zero = (val_feats == 0).float().mean().item() * 100
    val_relu = F.relu(val_feats)
    post_zero = (val_relu == 0).float().mean().item() * 100

    logger.info(
        "[%s] Val sparsity — pre-ReLU: %.2f%%  post-ReLU: %.2f%%",
        dataset_name, pre_zero, post_zero,
    )
    return F.relu(train_feats), val_relu


# ---------------------------------------------------------------------------
# 7. Linear probe training
# ---------------------------------------------------------------------------

def train_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    optim_preset: str = "L7",
    epochs: int = 100,
    batch_size: int = 512,
    device: str = "cuda",
) -> float:
    """
    Train a single ``nn.Linear`` on cached features and return top-1 accuracy.

    Presets
    ------
    L7  Adam, lr=1e-2, wd=0
    L9  SGD(momentum=0.9), lr=1e-2, wd=1e-6, cosine schedule
    """

    torch.manual_seed(0)
    np.random.seed(0)
    feat_dim = train_feats.shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.trunc_normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)

    if optim_preset == "L7":
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=1e-2, weight_decay=0,
        )
        scheduler = None
    elif optim_preset == "L9":
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-6,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown optim preset: {optim_preset}")

    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
    )

    # ---- training loop ----
    classifier.train()
    for _ in range(epochs):
        for feats_b, labels_b in train_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            logits = classifier(feats_b)
            loss = F.cross_entropy(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    # ---- evaluation ----
    classifier.eval()
    val_ds = TensorDataset(val_feats, val_labels)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for feats_b, labels_b in val_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            preds = classifier(feats_b).argmax(dim=1)
            correct += (preds == labels_b).sum().item()
            total += labels_b.size(0)

    acc = correct / total
    return acc


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Few-shot linear probe transfer evaluation on frozen SSL features",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to .pth or .ckpt SSL checkpoint",
    )
    parser.add_argument(
        "--model_name", type=str, default="vit_base_patch16_224.dino",
        help="timm model name (must match training architecture)",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=512,
        help="Projector output dimension (must match training config)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.01, 0.10, 1.0],
        help="Training-set fractions to evaluate (default: 1%% 10%% 100%%)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Probe training epochs (paper default: 100, use 1 for quick test)",
    )
    parser.add_argument(
        "--optim", type=str, default="L7", choices=["L7", "L9"],
        help="Optimizer preset — L7: Adam (Appendix L.7), L9: SGD+cosine (Appendix L.9)",
    )
    parser.add_argument(
        "--relu", action="store_true",
        help="Apply post-hoc ReLU to features before probing",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Probe batch size")
    parser.add_argument("--extract_batch_size", type=int, default=256, help="Feature extraction batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="linear-probe-transfer", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to checkpoint basename)")
    import wandb

    args = parser.parse_args()

    run_name = args.wandb_run_name or os.path.basename(args.checkpoint_path)
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),   # logs all hyperparameters automatically
    )

    # CUDA optimizations
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- load frozen model ----
    backbone,  feat_dim = load_model(
        args.checkpoint_path,
        args.model_name,
        args.proj_dim,
        args.device,
    )

    results: dict[str, dict[float, float]] = defaultdict(dict)

    for ds_name in args.datasets:
        ds_cfg = DATASETS[ds_name]
        logger.info("=" * 60)
        logger.info("Dataset: %s  (%d classes)", ds_name, ds_cfg["num_classes"])
        logger.info("=" * 60)

        # ---- extract & cache features once per dataset ----
        train_dataset = build_eval_dataset(ds_name, "train")
        val_dataset = build_eval_dataset(ds_name, "test")

        logger.info("Extracting train features")
        train_feats, train_labels = extract_features(
            backbone, train_dataset, args.device, args.extract_batch_size, args.num_workers,
        )
        logger.info("Extracting val/test features")
        val_feats, val_labels = extract_features(
            backbone, val_dataset, args.device, args.extract_batch_size, args.num_workers,
        )

        # ---- optional post-hoc ReLU ----
        if args.relu:
            train_feats, val_feats = apply_relu_and_report(
                train_feats, val_feats, ds_name,
            )

        # ---- probe under each data fraction ----
        for frac in sorted(args.fractions):
            sub_feats, sub_labels = stratified_subset(
                train_feats, train_labels, frac, seed=args.seed,
            )
            logger.info(
                "  fraction=%.0f%%  train_samples=%d", frac * 100, sub_feats.shape[0],
            )

            acc = train_linear_probe(
                sub_feats, sub_labels,
                val_feats, val_labels,
                num_classes=ds_cfg["num_classes"],
                optim_preset=args.optim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
            results[ds_name][frac] = acc
            n_shot = int(sub_labels.shape[0])   # actual number of training samples used
            wandb.log({
                f"eval/{ds_name}/top1_acc":  acc * 100,
                f"eval/{ds_name}/n_shot":    n_shot,
                f"eval/{ds_name}/fraction":  frac,
            }, step=None)   # step=None means all logged to same "summary" step
            logger.info("  -> Top-1 Accuracy: %.2f%%", acc * 100)

        # cleanup
        del train_dataset, val_dataset
        del train_feats, train_labels
        del val_feats, val_labels
        gc.collect()
        torch.cuda.empty_cache()

    # W&B Table
    fracs = sorted(args.fractions)
    col_names = ["dataset"] + [f"{int(f*100)}shot" for f in fracs]
    table = wandb.Table(columns=col_names)

    for ds_name in args.datasets:
        row = [ds_name] + [
            round(results[ds_name].get(f, float("nan")) * 100, 2)
            for f in fracs
        ]
        table.add_data(*row)

    wandb.log({"transfer_eval_table": table})

    # Also log each cell as a named summary scalar for easy sweep comparisons
    for ds_name in args.datasets:
        for frac in fracs:
            acc = results[ds_name].get(frac, float("nan"))
            wandb.summary[f"{ds_name}/{int(frac*100)}shot"] = round(acc * 100, 2)

    wandb.finish()

    for ds_name in args.datasets:
        row = f"{ds_name:<12}"
        for frac in fracs:
            acc = results[ds_name].get(frac, float("nan"))
            row += f"| {acc*100:6.2f}% "
        print(row)

    print("=" * 60)


if __name__ == "__main__":
    main()
