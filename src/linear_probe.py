"""
Few-Shot Linear Probe Transfer Evaluation on Frozen Image Embeddings.

Reproduces Table 2 of LeJEPA: frozen backbone features, K-shot linear probe
across 8 datasets, averaged over 3 seeds.

Feature extraction (consistent across all models and baselines):
  - Concatenation of CLS token from the last two transformer layers
  - For ViT without CLS token: average all patch tokens (standard practice)
  - LayerNorm on the concatenated features (DINO-style; improves probe performance)

Optimizer (consistent across all regimes):
  - AdamW, weight_decay=1e-6
  - LR schedule: same as pre-training — linear warmup (10%) + cosine annealing

Usage:
    python linear_probe.py \
        --checkpoint_path checkpoints/last.ckpt \
        --model_name vit_large_patch14_224.dino \
        --datasets dtd aircr cars cifar10 cifar100 flowers102 food101 pets \
        --k_shot 1 10 all \
        --seeds 0 1 2
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
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
        "hf_path": "HuggingFaceM4/FGVC-Aircraft",
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

# ---------------------------------------------------------------------------
# 2. Dynamic epoch schedule
#    Target gradient steps rather than a flat epoch count.
#    Full supervision uses 100 epochs unconditionally.
# ---------------------------------------------------------------------------

TARGET_STEPS = {1: 200, 10: 500}  # "all" always uses flat 100 epochs


def compute_epochs(n_train: int, batch_size: int, k) -> int:
    """Return epoch count so total gradient steps ≈ TARGET_STEPS[k]."""
    if k == "all":
        return 100
    steps_per_epoch = max(1, n_train // batch_size)
    return max(10, TARGET_STEPS[k] // steps_per_epoch)


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

EVAL_TRANSFORM = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
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
    k,                          # int or "all" — used for epochs + step count
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 0,
    lr: float = 1e-2,
) -> float:
    """
    Train a single nn.Linear on cached features.

    Consistent across all regimes:
      - AdamW, weight_decay=1e-6
      - LR schedule: linear warmup (10%) + cosine annealing (same as pre-training)
    """
    torch.manual_seed(seed)

    feat_dim = train_feats.shape[1]
    n_train = train_feats.shape[0]
    epochs = compute_epochs(n_train, batch_size, k)

    classifier = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.trunc_normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )

    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    classifier.train()
    step = 0
    for _ in range(epochs):
        for feats_b, labels_b in train_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            loss = F.cross_entropy(classifier(feats_b), labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

    # evaluation
    classifier.eval()
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for feats_b, labels_b in val_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            correct += (classifier(feats_b).argmax(dim=1) == labels_b).sum().item()
            total += labels_b.size(0)

    return correct / total


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
        "--k_shot", type=str, nargs="+", default=["1", "10", "all"],
        help="K values: integers for few-shot, 'all' for full supervision",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds to average over for k<all regimes",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2, help="Peak LR for probe (warmup + cosine)")
    parser.add_argument("--extract_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="lejepa-transfer-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    import wandb
    args = parser.parse_args()

    # parse k values — integers or the string "all"
    k_values = []
    for k in args.k_shot:
        k_values.append("all" if k == "all" else int(k))

    run_name = args.wandb_run_name or os.path.basename(args.checkpoint_path)
    wandb.init(
        project=args.wandb_project,
        entity="aho13-duke-university",
        name=run_name,
        config=vars(args),
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
                # full supervision: single run, no seed averaging needed
                acc = train_linear_probe(
                    train_feats, train_labels,
                    val_feats, val_labels,
                    num_classes=ds_cfg["num_classes"],
                    k=k,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=0,
                    lr=args.lr,
                )
                results[ds_name][k] = acc
                logger.info("  k=all -> Top-1: %.2f%%", acc * 100)
                wandb.log({f"{ds_name}/kall/seed0_acc": round(acc * 100, 2)})

            else:
                # few-shot: average over seeds
                seed_accs = []
                for seed in args.seeds:
                    sub_feats, sub_labels = k_shot_subset(
                        train_feats, train_labels, k, seed=seed,
                    )
                    logger.info(
                        "  k=%d  seed=%d  n_train=%d  epochs=%d",
                        k, seed, sub_feats.shape[0],
                        compute_epochs(sub_feats.shape[0], args.batch_size, k),
                    )
                    acc = train_linear_probe(
                        sub_feats, sub_labels,
                        val_feats, val_labels,
                        num_classes=ds_cfg["num_classes"],
                        k=k,
                        batch_size=args.batch_size,
                        device=args.device,
                        seed=seed,
                        lr=args.lr,
                    )
                    seed_accs.append(acc)
                    # log each individual seed result
                    wandb.log({
                        f"{ds_name}/k{k}/seed{seed}_acc": acc * 100,
                    })

                mean_acc = float(np.mean(seed_accs))
                std_acc = float(np.std(seed_accs))
                results[ds_name][k] = mean_acc
                logger.info(
                    "  k=%d -> mean=%.2f%%  std=%.2f%%  (seeds=%s)",
                    k, mean_acc * 100, std_acc * 100,
                    [f"{a*100:.2f}" for a in seed_accs],
                )
                wandb.summary[f"{ds_name}/k{k}_mean"] = round(mean_acc * 100, 2)
                wandb.summary[f"{ds_name}/k{k}_std"] = round(std_acc * 100, 2)

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
            label = "all" if k == "all" else f"{k}shot"
            logger.info("  %s: %.2f%%", label, avg)
            wandb.summary[f"avg/{label}"] = round(avg, 2)

    # ---- W&B tables: one per shot regime (rows=run_name, cols=datasets) ----
    for k in k_values:
        label = "all" if k == "all" else f"{k}shot"
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
    wandb.finish()

    # ---- console tables: one per shot regime (rows=run_name, cols=datasets) ----
    ds_cols = list(args.datasets)
    col_width = max(10, max(len(d) for d in ds_cols))
    run_width = max(14, len(run_name))

    for k in k_values:
        label = "all" if k == "all" else f"{k}shot"
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


if __name__ == "__main__":
    main()