"""
Within- and between-class trace statistics for embedding matrices (e.g. ImageNet).

Two entry points:

1. **From checkpoint (default):** load the timm backbone from a LeJEPA Lightning
   ``.ckpt``, extract **flat backbone** embeddings ``backbone(x)`` (same
   ``feat_dim`` as training ``all_emb`` / ``self.probe``), on an ImageNet-1K
   parquet or Hub split (default: **val**; official **test** has no labels).
   Load ``probe`` (``LayerNorm`` +
   ``Linear``) from the same checkpoint and report **top-1 accuracy** on valid
   labels in the same pass. Also applies ``encoder.proj`` when present and reports
   ``trace(Cov(z_proj))`` vs ``proj_dim`` (SIGReg sanity). Covariance on the
   backbone uses this same ``z`` as training ``all_emb`` (not
   ``linear_probe`` last-two-layer features).

2. **From files:** pass ``--z`` and ``--labels`` paths to precomputed arrays.

For each class y with enough samples, we form the class covariance C_y (unbiased,
matching ``numpy.cov``). The *within-class* scalar is the **unweighted** mean of
``trace(C_y)`` over classes with n_y >= 2.

The *between-class* scalar is ``trace(Cov(M))`` where rows of M are class means
``mu_y`` (one row per class with at least one sample).

**Sanity check (SIGReg / isotropy):** If pooled embeddings are approximately
``N(0, I)``, then ``trace(Cov(Z)) ≈ d``. The sum ``within + between`` uses the
**unweighted** definitions above and is **not** guaranteed to equal ``d`` or
``trace(Cov(Z))`` exactly; compare magnitudes rather than expecting equality.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

# Allow `import linear_probe` when running `python src/cov.py` from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def compute_within_between_traces(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Compute unweighted mean within-class trace and between-class trace of class means.

    Parameters
    ----------
    z
        Embeddings of shape ``(N, d)``.
    labels
        Integer class indices, shape ``(N,)``, in ``[0, num_classes-1]`` if
        ``num_classes`` is given; otherwise inferred as ``max(labels) + 1``.
    num_classes
        If set, loop ``y = 0 .. num_classes - 1``. Otherwise ``K = max(labels) + 1``.

    Returns
    -------
    dict with keys:
        ``within``, ``between``, ``sum_wb``, ``d``, ``N``, ``K``,
        ``num_classes_with_ge2`` (count used for within average),
        ``num_classes_with_ge1`` (rows used for between),
        ``trace_cov_total`` (``trace(Cov(Z))`` over all N points),
        ``per_class_n`` (length-K array of counts),
    """
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if z.shape[0] != len(labels):
        raise ValueError(f"z rows ({z.shape[0]}) must match labels ({len(labels)})")
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N, d), got shape {z.shape}")

    N, d = z.shape
    if num_classes is None:
        K = int(labels.max()) + 1
    else:
        K = int(num_classes)

    traces_w: list[float] = []
    mus: list[np.ndarray] = []
    per_class_n = np.zeros(K, dtype=np.int64)

    for y in range(K):
        idx = np.flatnonzero(labels == y)
        n_y = int(idx.size)
        per_class_n[y] = n_y
        if n_y == 0:
            mus.append(np.full(d, np.nan))
            continue
        zy = z[idx]
        mu_y = zy.mean(axis=0)
        mus.append(mu_y)
        if n_y >= 2:
            cy = np.cov(zy, rowvar=False)
            traces_w.append(float(np.trace(cy)))

    if not traces_w:
        within = float("nan")
    else:
        within = float(np.mean(traces_w))

    m = np.stack(mus, axis=0)
    valid = np.isfinite(m).all(axis=1)
    mu_valid = m[valid]
    n_between = int(mu_valid.shape[0])
    if n_between <= 1:
        between = float("nan")
    else:
        cov_between = np.cov(mu_valid, rowvar=False)
        between = float(np.trace(cov_between))

    sum_wb = within + between if np.isfinite(within) and np.isfinite(between) else float("nan")

    if N <= 1:
        trace_cov_total = float("nan")
    else:
        cov_all = np.cov(z, rowvar=False)
        trace_cov_total = float(np.trace(cov_all))

    return {
        "within": within,
        "between": between,
        "sum_wb": sum_wb,
        "d": d,
        "N": N,
        "K": K,
        "num_classes_with_ge2": len(traces_w),
        "num_classes_with_ge1": n_between,
        "trace_cov_total": trace_cov_total,
        "per_class_n": per_class_n,
    }


def _load_array(path: str) -> np.ndarray:
    path_lower = path.lower()
    if path_lower.endswith(".pt") or path_lower.endswith(".pth"):
        import torch

        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict) and "z" in t:
            t = t["z"]
        arr = t.detach().cpu().numpy() if hasattr(t, "numpy") else np.asarray(t)
    else:
        arr = np.load(path, allow_pickle=False)
    return np.asarray(arr, dtype=np.float64)


def _load_labels(path: str) -> np.ndarray:
    path_lower = path.lower()
    if path_lower.endswith(".pt") or path_lower.endswith(".pth"):
        import torch

        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict) and "labels" in t:
            t = t["labels"]
        arr = t.detach().cpu().numpy() if hasattr(t, "numpy") else np.asarray(t)
    else:
        arr = np.load(path, allow_pickle=False)
    return np.asarray(arr).reshape(-1).astype(np.int64, copy=False)


def _rel_err(x: float, target: float) -> float:
    if not np.isfinite(x) or target == 0:
        return float("nan")
    return abs(x - target) / abs(target)


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` prefixes injected by ``torch.compile``."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _probe_top1_on_arrays(
    z: np.ndarray,
    labels: np.ndarray,
    probe: Any,
    device: str,
    batch_size: int,
) -> float:
    """Top-1 accuracy of ``probe`` on frozen features ``z`` and integer ``labels``."""
    import torch

    probe.eval()
    n_cls = int(probe[1].out_features)
    correct = 0
    total = 0
    n = z.shape[0]
    for i in range(0, n, batch_size):
        zb = torch.from_numpy(z[i : i + batch_size]).to(device, dtype=torch.float32)
        lb = torch.from_numpy(labels[i : i + batch_size]).to(device, dtype=torch.long)
        with torch.no_grad():
            logits = probe(zb)
        pred = logits.argmax(dim=1)
        valid = (lb >= 0) & (lb < n_cls)
        correct += int((pred[valid] == lb[valid]).sum().item())
        total += int(valid.sum().item())
    if total <= 0:
        return float("nan")
    return float(correct) / float(total)


def _load_probe_from_lightning_checkpoint(
    checkpoint_path: str,
    feat_dim: int,
    device: str,
):
    """Load ``self.probe`` (LayerNorm + Linear) from a Lightning ``state_dict``."""
    import torch
    import torch.nn as nn

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        return None
    full_sd = _strip_compile_prefix(state["state_dict"])
    probe_sd: dict[str, Any] = {}
    for k, v in full_sd.items():
        if k.startswith("probe."):
            probe_sd[k[len("probe.") :]] = v
    if not probe_sd or "1.weight" not in probe_sd:
        return None
    if int(probe_sd["1.weight"].shape[1]) != feat_dim:
        raise ValueError(
            f"Probe in_features {probe_sd['1.weight'].shape[1]} != backbone feat_dim {feat_dim}"
        )
    num_classes = int(probe_sd["1.weight"].shape[0])
    probe = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, num_classes),
    )
    probe.load_state_dict(probe_sd, strict=True)
    probe.eval()
    probe.requires_grad_(False)
    probe.to(device)
    return probe



def _wandb_metrics(
    out_bb: dict[str, Any],
    *,
    out_proj: dict[str, Any] | None = None,
    proj_dim: float | None = None,
    probe_top1: float | None = None,
) -> dict[str, float]:
    """Scalar dict for wandb.log (finite values only where possible)."""
    d = float(out_bb["d"])
    sum_wb = out_bb["sum_wb"]
    ttot = out_bb["trace_cov_total"]
    m: dict[str, float] = {
        "cov/N": float(out_bb["N"]),
        "cov/d": d,
        "cov/K": float(out_bb["K"]),
        "cov/within_class_trace": float(out_bb["within"]),
        "cov/between_class_trace": float(out_bb["between"]),
        "cov/within_plus_between": float(sum_wb),
        "cov/trace_cov_total": float(ttot),
        "cov/num_classes_n_ge_2": float(out_bb["num_classes_with_ge2"]),
        "cov/num_classes_n_ge_1": float(out_bb["num_classes_with_ge1"]),
    }
    if probe_top1 is not None and np.isfinite(probe_top1):
        m["cov/probe_top1"] = float(probe_top1)
    re_sum = _rel_err(sum_wb, d)
    re_tot = _rel_err(ttot, d)
    if np.isfinite(re_sum):
        m["cov/rel_err_within_plus_between_vs_d"] = float(re_sum)
    if np.isfinite(re_tot):
        m["cov/rel_err_trace_cov_vs_d"] = float(re_tot)

    if out_proj is not None and proj_dim is not None and np.isfinite(proj_dim):
        dp = float(proj_dim)
        tp = float(out_proj["trace_cov_total"])
        m["cov_proj/N"] = float(out_proj["N"])
        m["cov_proj/d"] = dp
        m["cov_proj/K"] = float(out_proj["K"])
        m["cov_proj/within_class_trace"] = float(out_proj["within"])
        m["cov_proj/between_class_trace"] = float(out_proj["between"])
        m["cov_proj/within_plus_between"] = float(out_proj["sum_wb"])
        m["cov_proj/trace_cov_total"] = tp
        m["cov_proj/num_classes_n_ge_2"] = float(out_proj["num_classes_with_ge2"])
        m["cov_proj/num_classes_n_ge_1"] = float(out_proj["num_classes_with_ge1"])
        re_p = _rel_err(tp, dp)
        if np.isfinite(re_p):
            m["cov_proj/rel_err_trace_cov_vs_d"] = float(re_p)
        re_sump = _rel_err(float(out_proj["sum_wb"]), dp)
        if np.isfinite(re_sump):
            m["cov_proj/rel_err_within_plus_between_vs_d"] = float(re_sump)
    return m


def _print_report(
    out_bb: dict[str, Any],
    *,
    out_proj: dict[str, Any] | None = None,
    proj_dim: float | None = None,
    probe_top1: float | None = None,
) -> None:
    d = out_bb["d"]
    within = out_bb["within"]
    between = out_bb["between"]
    sum_wb = out_bb["sum_wb"]
    ttot = out_bb["trace_cov_total"]

    print("--- backbone ---")
    print(f"N = {out_bb['N']}, d = {d}, K (num_classes) = {out_bb['K']}")
    if probe_top1 is not None and np.isfinite(probe_top1):
        print(f"checkpoint probe top-1 (same z as cov): {probe_top1:.6g}")
    print(f"Classes with n_y >= 2 (within average): {out_bb['num_classes_with_ge2']}")
    print(f"Classes with n_y >= 1 (between rows): {out_bb['num_classes_with_ge1']}")
    print(f"within_class_trace (mean_y trace(C_y)): {within:.6g}")
    print(f"between_class_trace (trace Cov of mu_y's)): {between:.6g}")
    print(f"within + between = {sum_wb:.6g}")
    print(f"d (target if SIGReg isotropy): {d}")
    print(f"|(within+between) - d| / d = {_rel_err(sum_wb, float(d)):.6g}")
    print(f"trace(Cov(Z)) over all points: {ttot:.6g}")
    print(f"|trace(Cov(Z)) - d| / d = {_rel_err(ttot, float(d)):.6g}")

    if out_proj is not None and proj_dim is not None and np.isfinite(proj_dim):
        dp = float(proj_dim)
        tp = float(out_proj["trace_cov_total"])
        print("--- projection head (SIGReg sanity: trace(Cov) ~ proj_dim) ---")
        print(
            f"N = {out_proj['N']}, proj_dim = {dp}, K = {out_proj['K']}, "
            f"trace(Cov(z_proj)) = {tp:.6g}"
        )
        print(f"|trace(Cov(z_proj)) - proj_dim| / proj_dim = {_rel_err(tp, dp):.6g}")


def _extract_features_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str,
    proj_dim: int,
    split: str,
    dataset: str,
    imagenet1k_data_dir: str | None,
    inet100_data_dir: str | None,
    batch_size: int,
    num_workers: int,
    device: str,
    imagenet_source: str = "parquet",
    imagenet_hub_download_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load backbone (+ proj if present) and extract ``backbone(x)`` and ``proj(emb)``."""
    import torch
    from linear_probe import (
        build_imagenet1k_dataset,
        build_inet100_dataset,
        load_backbone_and_proj,
    )
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    backbone, proj, _feat_dim, _proj_dim_loaded = load_backbone_and_proj(
        checkpoint_path,
        model_name,
        proj_dim,
        device,
    )

    if dataset == "imagenet1k":
        ds = build_imagenet1k_dataset(
            split,
            imagenet1k_data_dir,
            source=imagenet_source,
            hub_download_mode=imagenet_hub_download_mode,
        )
    elif dataset == "inet100":
        ds = build_inet100_dataset(split, inet100_data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    all_feats: list = []
    all_proj: list = []
    all_labels: list = []

    for imgs, labels_b in tqdm(loader, desc="  backbone+proj"):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device, dtype=torch.bfloat16):
            emb = backbone(imgs)
        emb_f = emb.float()
        all_feats.append(emb_f.cpu())
        if proj is not None:
            with torch.no_grad():
                zp = proj(emb_f)
            all_proj.append(zp.cpu())
        all_labels.append(labels_b)

    z = torch.cat(all_feats, dim=0).numpy().astype(np.float64, copy=False)
    labels = torch.cat(all_labels, dim=0).numpy().reshape(-1).astype(np.int64, copy=False)
    z_proj: np.ndarray | None
    if proj is not None and all_proj:
        z_proj = torch.cat(all_proj, dim=0).numpy().astype(np.float64, copy=False)
    else:
        z_proj = None
    return z, z_proj, labels


def _filter_class_labels(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    z_proj: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep rows with labels in ``[0, num_classes)``; apply the same mask to ``z_proj``."""
    labels = np.asarray(labels).reshape(-1)
    valid = (labels >= 0) & (labels < num_classes)
    n_ok = int(np.sum(valid))
    if n_ok < 2:
        raise ValueError(
            f"Need at least 2 samples with labels in [0, {num_classes}); got {n_ok}. "
            "For ImageNet-1K test split, use --split val (test has label=-1)."
        )
    if n_ok < len(labels):
        print(
            f"Note: dropped {len(labels) - n_ok} rows with invalid/missing labels; using N={n_ok}."
        )
    zp_out = z_proj[valid] if z_proj is not None else None
    return z[valid], labels[valid], zp_out


DEFAULT_CHECKPOINT = (
    "data/checkpoints/LeJEPA_imagenet-1k/LV6_MV0_BS512_e100_ddp7/last.ckpt"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Within/between class trace decomposition for embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Lightning .ckpt path: load timm backbone and extract features on ImageNet-1K. "
            f"Default if --z/--labels omitted: {DEFAULT_CHECKPOINT}"
        ),
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "timm backbone name (must match checkpoint). Default: vit_large_patch16_224 for "
            "imagenet1k, vit_base_patch16_224 for inet100."
        ),
    )
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument(
        "--dataset",
        type=str,
        default="imagenet1k",
        choices=("imagenet1k", "inet100"),
        help="Eval dataset: ImageNet-1K parquet/Hub or ImageNet-100 parquet.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help=(
            "Split: val recommended. ImageNet-1K test has no labels. "
            "inet100: only train or val."
        ),
    )
    p.add_argument(
        "--imagenet_source",
        type=str,
        default="parquet",
        choices=("parquet", "hub"),
        help=(
            "parquet: local shards (default). hub: load_dataset(ILSVRC/imagenet-1k). "
            "Reloading the Hub does not add test labels (still -1)."
        ),
    )
    p.add_argument(
        "--imagenet_force_redownload",
        action="store_true",
        help="With --imagenet_source hub, pass download_mode=force_redownload to datasets.",
    )
    p.add_argument(
        "--imagenet1k_data_dir",
        type=str,
        default=None,
        help="Parquet root override when dataset=imagenet1k and imagenet_source=parquet.",
    )
    p.add_argument(
        "--inet100_data_dir",
        type=str,
        default=None,
        help="Parquet root override when dataset=inet100 (else INET100_PARQUET_DIR).",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--z", default=None, help="Path to .npy or .pt tensor (N, d) [file mode]")
    p.add_argument("--labels", default=None, help="Path to .npy or .pt tensor (N,) [file mode]")
    p.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Explicit K (default: 1000 / 100 by dataset in checkpoint mode; else max(labels)+1)",
    )
    p.add_argument(
        "--save_z",
        type=str,
        default=None,
        help="Optional path to save extracted z as .npy (checkpoint mode)",
    )
    p.add_argument(
        "--save_labels",
        type=str,
        default=None,
        help="Optional path to save labels as .npy (checkpoint mode)",
    )
    p.add_argument(
        "--no_probe_eval",
        action="store_true",
        help="Do not load checkpoint `probe` or report top-1 (covariance only).",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    p.add_argument("--wandb_project", default="Covariance_Tests")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument(
        "--wandb_tags",
        default=None,
        help="Comma-separated tags for the wandb run",
    )
    args = p.parse_args(argv)

    if args.model_name is None:
        args.model_name = (
            "vit_base_patch16_224" if args.dataset == "inet100" else "vit_large_patch16_224"
        )

    if args.dataset == "inet100" and args.split not in ("train", "val"):
        p.error("dataset=inet100 only supports --split train or val")

    file_mode = args.z is not None or args.labels is not None
    if file_mode and (args.z is None or args.labels is None):
        p.error("File mode requires both --z and --labels")

    z_proj: np.ndarray | None = None
    out_proj: dict[str, Any] | None = None
    proj_dim_report: float | None = None

    if not file_mode:
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT
        ckpt = os.path.normpath(ckpt)
        if not os.path.isfile(ckpt):
            p.error(f"Checkpoint not found: {ckpt}")
        hub_mode = "force_redownload" if args.imagenet_force_redownload else None
        print(
            f"Loading from {ckpt}  (dataset={args.dataset}, split={args.split}, "
            f"imagenet_source={args.imagenet_source})"
        )
        z, z_proj, labels = _extract_features_from_checkpoint(
            ckpt,
            model_name=args.model_name,
            proj_dim=args.proj_dim,
            split=args.split,
            dataset=args.dataset,
            imagenet1k_data_dir=args.imagenet1k_data_dir,
            inet100_data_dir=args.inet100_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            imagenet_source=args.imagenet_source,
            imagenet_hub_download_mode=hub_mode,
        )
        if z_proj is None:
            print(
                "Warning: no projector weights in checkpoint; skipping projection SIGReg metrics.",
            )
        default_k = 1000 if args.dataset == "imagenet1k" else 100
        num_classes = args.num_classes if args.num_classes is not None else default_k
        z, labels, z_proj = _filter_class_labels(
            z,
            labels,
            num_classes=num_classes,
            z_proj=z_proj,
        )
        if z_proj is not None:
            proj_dim_report = float(z_proj.shape[1])

        probe_top1: float | None = None
        if not args.no_probe_eval:
            probe = _load_probe_from_lightning_checkpoint(ckpt, int(z.shape[1]), args.device)
            if probe is None:
                print(
                    "Warning: no `probe.*` weights in checkpoint; skipping probe top-1.",
                )
            else:
                probe_top1 = _probe_top1_on_arrays(
                    z,
                    labels,
                    probe,
                    args.device,
                    args.batch_size,
                )
        if args.save_z:
            np.save(args.save_z, z)
            print(f"Saved z -> {args.save_z}")
        if args.save_labels:
            np.save(args.save_labels, labels)
            print(f"Saved labels -> {args.save_labels}")
        file_probe_top1 = probe_top1
    else:
        z = _load_array(args.z)
        labels = _load_labels(args.labels)
        num_classes = args.num_classes
        file_probe_top1 = None

    out_bb = compute_within_between_traces(z, labels, num_classes=num_classes)
    if z_proj is not None and proj_dim_report is not None:
        out_proj = compute_within_between_traces(
            z_proj,
            labels,
            num_classes=num_classes,
        )

    _print_report(
        out_bb,
        out_proj=out_proj,
        proj_dim=proj_dim_report,
        probe_top1=file_probe_top1,
    )

    if not args.no_wandb:
        import wandb

        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        config = {
            "z_path": args.z,
            "labels_path": args.labels,
            "num_classes": num_classes,
            "checkpoint": None if file_mode else (args.checkpoint or DEFAULT_CHECKPOINT),
            "split": None if file_mode else args.split,
            "model_name": None if file_mode else args.model_name,
            "dataset": None if file_mode else args.dataset,
            "imagenet_source": None if file_mode else args.imagenet_source,
            "imagenet_force_redownload": None
            if file_mode
            else bool(args.imagenet_force_redownload),
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=tags,
            config=config,
        )
        try:
            wandb.log(
                _wandb_metrics(
                    out_bb,
                    out_proj=out_proj,
                    proj_dim=proj_dim_report,
                    probe_top1=file_probe_top1,
                )
            )
        finally:
            wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
