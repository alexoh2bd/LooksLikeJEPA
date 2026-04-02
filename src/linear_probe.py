"""
Linear probe transfer evaluation on frozen backbone features (LeJEPA-style).

Label regimes (fraction of *labeled training* data, stratified per class):
  - ``1``  → 1%   (paper: "1-shot" naming)
  - ``10`` → 10%  ("10-shot")
  - ``all`` → 100% ("all-shot")

Probe training (paper): 100 epochs, Adam, learning rate :math:`10^{-2}`, batch size
512, **no weight decay**. No data augmentation during probing.

Evaluation images: resize so the **shorter side is 256**, center-crop to 224×224,
ImageNet mean/std normalization (no random augmentation).

Feature extraction (frozen backbone):
  - Concatenate CLS from the last two transformer layers (or mean patch if no CLS)
  - LayerNorm on the concatenated features

Usage:
    python linear_probe.py \
        --checkpoint_path data/checkpoints/<run>/last.ckpt \
        --model_name vit_large_patch14_224.dino \
        --datasets dtd aircr cars cifar10 cifar100 flowers102 food101 pets \
        --label_regimes 1 10 all \
        --seeds 0 1 2

Optional ImageNet-1K val top-1 (full train) unless ``--skip_imagenet1k_full``.
"""

import gc
import argparse
import glob
import logging
from collections import defaultdict
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def save_prefix_from_checkpoint_path(checkpoint_path: str) -> str:
    """Folder containing the checkpoint, as a posix-style path.

    Matches ``run_training_loop``'s ``save_prefix`` when the checkpoint lives
    under ``<cwd>/data/checkpoints/<save_prefix>/last.ckpt`` (or
    ``<cwd>/checkpoints/...``). Otherwise falls back to a path relative to cwd,
    then to the absolute parent directory.
    """
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    cwd = os.getcwd()
    for root_name in ("data/checkpoints", "checkpoints"):
        root = os.path.normpath(os.path.join(cwd, root_name))
        parent_n = os.path.normpath(parent)
        try:
            common = os.path.commonpath([root, parent_n])
        except ValueError:
            continue
        if common == root and (parent_n == root or parent_n.startswith(root + os.sep)):
            rel = os.path.relpath(parent_n, root)
            return rel.replace(os.sep, "/")
    try:
        rel = os.path.relpath(parent, cwd)
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    except ValueError:
        pass
    return parent.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# 1. Dataset registry — 8 datasets matching Table 2
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
    "aircr": {
        "hf_path": "mteb/FGVCAircraft",
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
}

# ImageNet-1K (ILSVRC): local parquet layout matches ``src/ds.HFDataset`` / training pipeline.
IMAGENET1K_NUM_CLASSES = 1000

# Paper: linear probe always trained for 100 epochs.
PROBE_EPOCHS = 100

# CLI integers 1 and 10 denote 1% and 10% of stratified training labels (not k-shot counts).
LABEL_FRAC = {1: 0.01, 10: 0.10}


def log_key(k) -> str:
    """W&B / summary suffix: k1, k10, kall."""
    if k == "all":
        return "kall"
    return f"k{k}"

# ---------------------------------------------------------------------------
# 3. Model loading — backbone only for cross-dataset transfer
# ---------------------------------------------------------------------------

def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` prefixes injected by ``torch.compile``."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_model(
    checkpoint_path: str,
    model_name: str = "vit_large_patch14_224.dino",
    proj_dim: int = 512,
    device: str = "cuda",
):
    """
    Load frozen backbone from checkpoint.
    Returns (backbone, feat_dim) — frozen and on *device*.

    For cross-dataset transfer we use raw backbone features only,
    not projected features, since the projector head is trained on
    ImageNet-1K statistics and does not generalize across domains.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "backbone_only" in state:
        # standalone backbone dict saved directly
        backbone_sd = _strip_compile_prefix(state["backbone_only"])

    elif "encoder" in state:
        enc_sd = _strip_compile_prefix(state["encoder"])
        backbone_sd = {
            k.replace("backbone.", "", 1): v
            for k, v in enc_sd.items()
            if k.startswith("backbone.")
        }

    elif "state_dict" in state:
        # Lightning checkpoint — full module state dict
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

    backbone.eval()
    backbone.requires_grad_(False)
    backbone.to(device)

    logger.info(
        "Loaded backbone from %s  (feat_dim=%d)", checkpoint_path, feat_dim
    )
    return backbone, feat_dim


# ---------------------------------------------------------------------------
# 4. Eval-mode image dataset
# ---------------------------------------------------------------------------

# Shorter side → 256, then center 224×224; ImageNet stats; no train-time augmentation.
EVAL_TRANSFORM = v2.Compose([
    v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.bfloat16, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
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
    cfg = DATASETS[dataset_name]
    split_str = cfg["train_split"] if split == "train" else cfg["test_split"]
    hf_ds = load_dataset(cfg["hf_path"], split=split_str, trust_remote_code=True)
    logger.info("Loaded %s split='%s' (%d samples)", dataset_name, split_str, len(hf_ds))
    return ImageDataset(hf_ds, cfg["image_key"], cfg["label_key"])


def _default_imagenet1k_parquet_dir() -> str:
    return os.path.join(
        os.getcwd(),
        "data/hub/datasets--ILSVRC--imagenet-1k/snapshots/"
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data",
    )


def build_imagenet1k_dataset(split: str, data_dir: str | None = None) -> ImageDataset:
    """Load ImageNet-1K from local parquet shards (same source as pretraining).

    ``split`` is ``train``, ``val`` (ILSVRC validation, 50k labeled), or ``test``.
    Official competition test labels are not public; for a standard top-1 number,
    use ``val``. Use ``test`` only if your parquet includes ``label`` columns.
    """
    root = data_dir or os.environ.get(
        "IMAGENET1K_PARQUET_DIR", _default_imagenet1k_parquet_dir()
    )
    patterns = {
        "train": "train*.parquet",
        "val": "validation*.parquet",
        "test": "test*.parquet",
    }
    if split not in patterns:
        raise ValueError(f"split must be train, val, or test; got {split!r}")
    files = sorted(glob.glob(os.path.join(root, patterns[split])))
    if not files:
        raise FileNotFoundError(
            f"No parquet files for split={split!r} under {root} (pattern {patterns[split]})"
        )
    hf_ds = load_dataset("parquet", data_files=files, split="train")
    image_key = "image" if "image" in hf_ds.column_names else "img"
    logger.info(
        "Loaded ImageNet-1K split=%s (%d samples) from %s",
        split,
        len(hf_ds),
        root,
    )
    return ImageDataset(hf_ds, image_key, "label")


# ---------------------------------------------------------------------------
# 5. Feature extraction — last-2-layer CLS concat + LayerNorm
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
    Extract frozen backbone features.

    For ViT: concatenate CLS token from last two layers, apply LayerNorm.
    For ViT without CLS: average all patch tokens per layer, then concat.
    For non-ViT (e.g. ConvNeXt): standard forward output.
    """
    use_last_two = hasattr(backbone, "blocks") and len(backbone.blocks) >= 2
    if use_last_two:
        feat_dim = 2 * backbone.num_features
        layer_norm = nn.LayerNorm(feat_dim).to(device)
    else:
        feat_dim = backbone.num_features
        layer_norm = None

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
            emb = _get_last_two_layer_features(backbone, imgs, device)
        if layer_norm is not None:
            emb = layer_norm(emb.float())
        all_feats.append(emb.float().cpu())
        all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logger.info(
        "  cached %d features of dim %d (last-2-layer=%s)",
        features.shape[0], features.shape[1], use_last_two,
    )
    return features, labels


# ---------------------------------------------------------------------------
# 6. K-shot subsetting — no global seed mutation
# ---------------------------------------------------------------------------

def k_shot_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    seed: int = 0,
):
    """
    Select exactly k samples per class.
    Uses a local numpy Generator — does NOT mutate global random state.
    """
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    indices = []
    for cls in np.unique(labels_np):
        cls_idx = np.where(labels_np == cls)[0]
        chosen = rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
        indices.append(chosen)
    idx = torch.from_numpy(np.concatenate(indices))
    return features[idx], labels[idx]


# ---------------------------------------------------------------------------
# 7. Linear probe
# ---------------------------------------------------------------------------

def train_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 0,
    lr: float = 1e-2,
    epochs: int = PROBE_EPOCHS,
) -> float:
    """Adam, lr=1e-2, no weight decay, fixed epoch count (paper: 100)."""
    torch.manual_seed(seed)

    classifier = nn.Linear(train_feats.shape[1], num_classes).to(device)
    nn.init.trunc_normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=0.0)

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    classifier.train()
    for _ in range(epochs):
        for feats_b, labels_b in train_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            loss = F.cross_entropy(classifier(feats_b), labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    classifier.eval()
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels),
        batch_size=batch_size,
    )
    correct, total = 0, 0
    with torch.no_grad():
        for feats_b, labels_b in val_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            correct += (classifier(feats_b).argmax(dim=1) == labels_b).sum().item()
            total += labels_b.size(0)
    return correct / total


def fraction_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
    seed: int = 0,
):
    """Stratified subsample: per class, take ``floor(n_c * fraction)`` points (paper-style).

    For fractions below 1, each class keeps at least one example when possible
    (``min(n_c, max(1, int(n_c * fraction)))``) so every class is represented.
    ``fraction == 1.0`` uses all training indices for that class.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    indices = []
    for cls in np.unique(labels_np):
        cls_idx = np.where(labels_np == cls)[0]
        n_c = len(cls_idx)
        if fraction >= 1.0:
            n_select = n_c
        else:
            n_select = min(n_c, max(1, int(n_c * fraction)))
        chosen = rng.choice(cls_idx, size=n_select, replace=False)
        indices.append(chosen)
    idx = torch.from_numpy(np.concatenate(indices))
    return features[idx], labels[idx]

def _get_last_two_layer_features(backbone, x, device):
    """
    Extract CLS (or mean patch) from last two blocks, concatenate.
    Returns raw concatenated features (B, 2*D) for ViT; (B, D) for non-ViT.
    Caller applies LayerNorm after concatenation.
    """
    if not hasattr(backbone, "blocks") or len(backbone.blocks) < 2:
        # Non-ViT (e.g. ConvNeXt): fallback to standard forward
        return backbone(x)

    captured = []

    def make_hook(idx):
        def hook(module, inp, out):
            captured.append((idx, out.detach()))

        return hook

    hooks = [
        backbone.blocks[-2].register_forward_hook(make_hook(0)),
        backbone.blocks[-1].register_forward_hook(make_hook(1)),
    ]
    try:
        _ = backbone(x)
    finally:
        for h in hooks:
            h.remove()

    captured.sort(key=lambda t: t[0])
    feats = [captured[0][1], captured[1][1]]  # (B, N+1, D) each

    has_cls = getattr(backbone, "cls_token", None) is not None
    if has_cls:
        layer_feats = [f[:, 0] for f in feats]  # CLS token
    else:
        layer_feats = [f.mean(dim=1) for f in feats]  # mean patch (standard for ViT w/o CLS)

    return torch.cat(layer_feats, dim=1)  # (B, 2*D)

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, default="vit_large_patch14_224.dino",
        help="timm model name — must match pretraining architecture",
    )
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--label_regimes",
        "--k_shot",
        dest="label_regimes",
        type=str,
        nargs="+",
        default=["1", "10", "all"],
        help="1 → 1%% train, 10 → 10%% train, all → 100%% (stratified per class). "
        "Alias: --k_shot (legacy name).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds to average over for k<all regimes",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam LR for linear probe (paper: 1e-2, no schedule).",
    )
    parser.add_argument("--extract_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="lejepa-transfer-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--skip_imagenet1k_full",
        action="store_true",
        help="Skip ImageNet-1K full train → eval linear probe at the end.",
    )
    parser.add_argument(
        "--imagenet1k_data_dir",
        type=str,
        default=None,
        help="Parquet root for ILSVRC shards (train/validation/test). "
        "Default: IMAGENET1K_PARQUET_DIR env or data/hub/.../imagenet-1k/.../data.",
    )
    parser.add_argument(
        "--imagenet1k_eval_split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Eval split after training on full train: val (standard 50k labeled top-1) "
        "or test (only if parquet includes labels).",
    )
    import wandb
    args = parser.parse_args()

    # Regimes: 1 → 1%, 10 → 10%, all → 100% of labeled training data
    k_values: list = []
    for r in args.label_regimes:
        k_values.append("all" if r == "all" else int(r))

    save_prefix = save_prefix_from_checkpoint_path(args.checkpoint_path)
    run_name = args.wandb_run_name or save_prefix
    wandb_cfg = dict(vars(args))
    wandb_cfg["save_prefix"] = save_prefix
    logger.info(
        "wandb run name=%s  (save_prefix from checkpoint dir; override with --wandb_run_name)",
        run_name,
    )
    wandb.init(
        project=args.wandb_project,
        entity="aho13-duke-university",
        name=run_name,
        config=wandb_cfg,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    backbone, feat_dim = load_model(
        args.checkpoint_path, args.model_name, args.proj_dim, args.device,
    )

    # results[dataset][k] = mean accuracy across seeds
    results: dict[str, dict] = defaultdict(dict)

    for ds_name in args.datasets:
        ds_cfg = DATASETS[ds_name]
        logger.info("=" * 60)
        logger.info("Dataset: %s  (%d classes)", ds_name, ds_cfg["num_classes"])
        logger.info("=" * 60)

        train_feats, train_labels = extract_features(
            backbone, build_eval_dataset(ds_name, "train"),
            args.device, args.extract_batch_size, args.num_workers,
        )
        val_feats, val_labels = extract_features(
            backbone, build_eval_dataset(ds_name, "test"),
            args.device, args.extract_batch_size, args.num_workers,
        )

        for k in k_values:
            if k == "all":
                # 100% of labeled training data
                acc = train_linear_probe(
                    train_feats,
                    train_labels,
                    val_feats,
                    val_labels,
                    num_classes=ds_cfg["num_classes"],
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=0,
                    lr=args.lr,
                    epochs=PROBE_EPOCHS,
                )
                results[ds_name][k] = acc
                tag = log_key(k)
                logger.info("  %s (100%% train) -> Top-1: %.2f%%", tag, acc * 100)
                wandb.log({f"{ds_name}/{tag}/seed0_acc": round(acc * 100, 2)})

            else:
                if k not in LABEL_FRAC:
                    raise ValueError(
                        f"Unsupported regime {k!r}; use 1 (1%%), 10 (10%%), or all (100%%)."
                    )
                frac = LABEL_FRAC[k]
                tag = log_key(k)
                seed_accs = []
                for seed in args.seeds:
                    sub_feats, sub_labels = fraction_subset(
                        train_feats, train_labels, frac, seed=seed,
                    )
                    logger.info(
                        "  %s (frac=%.4f) seed=%d  n_train=%d  probe_epochs=%d",
                        tag,
                        frac,
                        seed,
                        sub_feats.shape[0],
                        PROBE_EPOCHS,
                    )
                    acc = train_linear_probe(
                        sub_feats,
                        sub_labels,
                        val_feats,
                        val_labels,
                        num_classes=ds_cfg["num_classes"],
                        batch_size=args.batch_size,
                        device=args.device,
                        seed=seed,
                        lr=args.lr,
                        epochs=PROBE_EPOCHS,
                    )
                    seed_accs.append(acc)
                    wandb.log({
                        f"{ds_name}/{tag}/seed{seed}_acc": acc * 100,
                    })

                mean_acc = float(np.mean(seed_accs))
                std_acc = float(np.std(seed_accs))
                results[ds_name][k] = mean_acc
                logger.info(
                    "  %s -> mean=%.2f%%  std=%.2f%%  (seeds=%s)",
                    tag,
                    mean_acc * 100,
                    std_acc * 100,
                    [f"{a*100:.2f}" for a in seed_accs],
                )
                wandb.summary[f"{ds_name}/{tag}_mean"] = round(mean_acc * 100, 2)
                wandb.summary[f"{ds_name}/{tag}_std"] = round(std_acc * 100, 2)

        del train_feats, train_labels, val_feats, val_labels
        gc.collect()
        torch.cuda.empty_cache()

    # ---- aggregate across datasets (mirrors Table 2 avg column) ----
    logger.info("=" * 60)
    logger.info("AGGREGATE (mean across %d datasets)", len(args.datasets))
    for k in k_values:
        per_ds = [results[ds][k] for ds in args.datasets if k in results[ds]]
        if per_ds:
            avg = float(np.mean(per_ds)) * 100
            label = log_key(k)
            logger.info("  %s: %.2f%%", label, avg)
            wandb.summary[f"avg/{label}"] = round(avg, 2)

    # ---- W&B tables: one per shot regime (rows=run_name, cols=datasets) ----
    for k in k_values:
        label = log_key(k)
        col_names = ["run_name"] + list(args.datasets) + ["avg"]
        table = wandb.Table(columns=col_names)
        row_vals = [run_name]
        for ds_name in args.datasets:
            acc = results[ds_name].get(k, float("nan"))
            row_vals.append(round(acc * 100, 2) if not np.isnan(acc) else float("nan"))
        per_ds = [results[ds][k] for ds in args.datasets if k in results[ds]]
        avg = float(np.mean(per_ds)) * 100 if per_ds else float("nan")
        row_vals.append(round(avg, 2) if not np.isnan(avg) else float("nan"))
        table.add_data(*row_vals)
        wandb.log({f"transfer_eval_{label}": table})

    # ---- ImageNet-1K: full linear probe (train on full train, eval on val or test) ----
    imagenet1k_top1 = None
    imagenet1k_eval_split_used: str | None = None
    if not args.skip_imagenet1k_full:
        inet_root = args.imagenet1k_data_dir or os.environ.get(
            "IMAGENET1K_PARQUET_DIR", _default_imagenet1k_parquet_dir()
        )
        if not os.path.isdir(inet_root):
            logger.warning(
                "ImageNet-1K parquet dir not found (%s); skipping full probe.",
                inet_root,
            )
        else:
            eval_split = args.imagenet1k_eval_split
            try:
                logger.info("=" * 60)
                logger.info(
                    "ImageNet-1K: full linear probe (train on train, eval on %s)",
                    eval_split,
                )
                eval_ds = build_imagenet1k_dataset(eval_split, args.imagenet1k_data_dir)
                row0 = eval_ds.ds[0]
                lab0 = row0.get(eval_ds.label_key)
                if eval_split == "test" and (
                    lab0 is None
                    or (isinstance(lab0, (int, np.integer)) and int(lab0) < 0)
                ):
                    logger.warning(
                        "ImageNet-1K test split has no usable labels; using val."
                    )
                    eval_split = "val"
                    eval_ds = build_imagenet1k_dataset("val", args.imagenet1k_data_dir)

                train_ds_inet = build_imagenet1k_dataset("train", args.imagenet1k_data_dir)
                inet_train_feats, inet_train_labels = extract_features(
                    backbone,
                    train_ds_inet,
                    args.device,
                    args.extract_batch_size,
                    args.num_workers,
                )
                inet_eval_feats, inet_eval_labels = extract_features(
                    backbone,
                    eval_ds,
                    args.device,
                    args.extract_batch_size,
                    args.num_workers,
                )
                imagenet1k_top1 = train_linear_probe(
                    inet_train_feats,
                    inet_train_labels,
                    inet_eval_feats,
                    inet_eval_labels,
                    num_classes=IMAGENET1K_NUM_CLASSES,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=0,
                    lr=args.lr,
                    epochs=PROBE_EPOCHS,
                )
                imagenet1k_eval_split_used = eval_split
                tag = "val_top1" if eval_split == "val" else "test_top1"
                logger.info(
                    "  ImageNet-1K top-1 (eval=%s): %.2f%%",
                    eval_split,
                    imagenet1k_top1 * 100,
                )
                wandb.log({f"imagenet1k/{tag}": round(imagenet1k_top1 * 100, 2)})
                wandb.summary[f"imagenet1k/{tag}"] = round(imagenet1k_top1 * 100, 2)
                del (
                    inet_train_feats,
                    inet_train_labels,
                    inet_eval_feats,
                    inet_eval_labels,
                )
                gc.collect()
                torch.cuda.empty_cache()
            except FileNotFoundError as e:
                logger.warning("ImageNet-1K full probe skipped: %s", e)

    wandb.finish()

    # ---- console tables: one per shot regime (rows=run_name, cols=datasets) ----
    ds_cols = list(args.datasets)
    col_width = max(10, max(len(d) for d in ds_cols))
    run_width = max(14, len(run_name))

    for k in k_values:
        label = log_key(k)
        print(f"\n{'=' * 60}")
        print(f"  {label.upper()} TABLE")
        print("=" * 60)
        header = f"{'run_name':<{run_width}}" + "".join(
            f"| {d:>{col_width}} " for d in ds_cols
        ) + f"| {'avg':>{col_width}} "
        print(header)
        print("-" * len(header))
        row_str = f"{run_name:<{run_width}}"
        for ds_name in args.datasets:
            acc = results[ds_name].get(k, float("nan"))
            val = f"{acc * 100:.2f}%" if not np.isnan(acc) else "nan"
            row_str += f"| {val:>{col_width}} "
        per_ds = [results[ds][k] for ds in args.datasets if k in results[ds]]
        avg = float(np.mean(per_ds)) * 100 if per_ds else float("nan")
        avg_val = f"{avg:.2f}%" if not np.isnan(avg) else "nan"
        row_str += f"| {avg_val:>{col_width}} "
        print(row_str)
        print("=" * len(header))

    if imagenet1k_top1 is not None and imagenet1k_eval_split_used is not None:
        print(f"\n{'=' * 60}")
        print("  IMAGENET-1K FULL LINEAR PROBE (train on train)")
        print("=" * 60)
        print(
            f"  eval={imagenet1k_eval_split_used}  top-1: {imagenet1k_top1 * 100:.2f}%"
        )
        print("=" * 60)


if __name__ == "__main__":
    main()