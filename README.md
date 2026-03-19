# Looks Like JEPA!

Human visual recognition is inherently relational: we identify a cheetah partly because we've already learned what a cat looks like, drawing on similarity between previously encountered objects to make sense of new ones. Standard self-supervised methods like JEPA learn representations from multiple augmented views of a single image, but never explicitly encourage the model to connect representations across semantically related images during pretraining.
This project investigates whether semantically guided view selection — exposing the model to local views drawn from related but distinct images — produces representations that transfer more effectively to unseen tasks. We explore two strategies:

Semantic cross-instance views (SCV): local crops sampled from a different image sharing the same class label as the anchor.
kP-views (neighbor views): local crops sampled from the anchor's top-k nearest neighbors in a pretrained embedding space (e.g., CLIP), requiring no labels at all.

Both strategies augment the standard LeJEPA training loop: the predictor must reconstruct patch-level representations of views that may originate from a different image than the anchor, encouraging the encoder to build representations grounded in shared semantics rather than low-level pixel statistics.
We evaluate representation quality through few-shot linear transfer across multiple classification benchmarks (DTD, CIFAR-10/100, Flowers-102, Food-101, Oxford Pets, and others), measuring how well the learned features generalize to data and tasks the model has never seen during pretraining.

**W&B project:** [VIT_JEPA_Views](https://wandb.ai/aho13-duke-university/VIT_JEPA_Views)

---

## Table of Contents

1. [Installation](#installation)
2. [Data Setup](#data-setup)
3. [Training — Single GPU](#training--single-gpu)
4. [Training — Multi-GPU DDP](#training--multi-gpu-ddp)
5. [Few-Shot Linear Probe Evaluation](#few-shot-linear-probe-evaluation)
6. [Key Hyperparameters](#key-hyperparameters)
7. [Directory Structure](#directory-structure)

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up environment
git clone <repo-url> jepa_tests
cd jepa_tests
uv sync
```

All training and evaluation commands should be run with `uv run` (or after activating the venv with `source .venv/bin/activate`).

---

## Data Setup

Data is loaded from local parquet shards via HuggingFace `datasets`. Set `HF_DATASETS_OFFLINE=1` to prevent network calls once data is cached.

### ImageNet-1K

Expected path: `data/hub/datasets--ILSVRC--imagenet-1k/snapshots/<hash>/data/`

```
data/hub/datasets--ILSVRC--imagenet-1k/snapshots/<hash>/data/
├── train-00000-of-01024.parquet
├── ...
├── validation-00000-of-00128.parquet
└── ...
```

### ImageNet-100 (inet100)

Expected path: `data/cache/datasets--clane9--imagenet-100/snapshots/<hash>/data/`

```
data/cache/datasets--clane9--imagenet-100/snapshots/<hash>/data/
├── train-*.parquet
└── validation*.parquet
```

---

## Training — Single GPU

Launch via SLURM:

```bash
sbatch scripts/train.sh
```

Or run directly:

```bash
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

uv run src/run_training_loop.py \
  +reg=LeJEPA \
  +model_name=vit_large_patch16_224 \
  +dataset=imagenet-1k \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=1e-2 \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +V_mixed=2 \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +grad_accum=1 \
  +num_workers=16 \
  +prefetch_factor=2 \
  +distributed=False \
  +world_size=1 \
  +seed=0 \
  +log_interval=40
```

**inet100 equivalent** (faster iteration):

```bash
uv run src/run_training_loop.py \
  +reg=LeJEPA \
  +model_name=vit_large_patch16_224 \
  +dataset=inet100 \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=1e-2 \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +V_mixed=2 \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +num_workers=8 \
  +distributed=False \
  +world_size=1 \
  +seed=0
```

### Checkpoint Resume

Training resumes automatically from `data/checkpoints/<run_name>/last.ckpt` if it exists. The run name is derived from the method and hyperparameters:

```
data/checkpoints/<reg>_<dataset>/LV<V_local>_MV<V_mixed>_BS<bs>_e<epochs>/
├── last.ckpt
├── epoch=N-val_acc=0.XXX.ckpt   # top-2 by val/acc
└── ...
```

---

## Training — Multi-GPU DDP

Launch via SLURM:

```bash
sbatch scripts/train_ddp.sh
```

Or run directly (NCCL backend, Lightning DDP):

```bash
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4

# 4 GPUs, 1 node
srun --ntasks=4 --ntasks-per-node=4 uv run src/run_training_loop.py \
  +reg=hybrid \
  +model_name=vit_large_patch16_224 \
  +dataset=imagenet-1k \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=1e-2 \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=6 \
  +V_mixed=0 \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +grad_accum=1 \
  +num_workers=7 \
  +prefetch_factor=2 \
  +distributed=True \
  +world_size=4 \
  +num_nodes=1 \
  +seed=0 \
  +log_interval=40
```

**inet100 equivalent (4 GPUs):**

```bash
srun --ntasks=4 --ntasks-per-node=4 uv run src/run_training_loop.py \
  +reg=hybrid \
  +model_name=vit_large_patch16_224 \
  +dataset=inet100 \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=1e-2 \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=6 \
  +V_mixed=0 \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +num_workers=7 \
  +distributed=True \
  +world_size=4 \
  +num_nodes=1 \
  +seed=0
```

> **Note:** `bs` is the **total effective batch size** across all GPUs. Each GPU receives `bs // world_size` samples per step. With `world_size=4` and `bs=512`, each GPU processes 128 samples.

### SLURM Resource Guide

| Setup | `--nodes` | `--ntasks-per-node` | `--cpus-per-task` | `--gres` |
|---|---|---|---|---|
| 4× a6000 (1 node) | 1 | 4 | 8 | `gpu:a6000:4` |
| 8× a6000 (2 nodes) | 2 | 4 | 6 | `gpu:a6000:4` |

For cross-node (2+ nodes) jobs, ensure the inter-node network supports NCCL. The NCCL timeout is 30 minutes by default.

---

## Few-Shot Linear Probe Evaluation

Evaluates a frozen pretrained backbone on six downstream datasets under 1%, 10%, and 100% data regimes. Results are logged to W&B.

### Available Datasets

`dtd`, `cifar10`, `cifar100`, `flowers102`, `food101`, `pets`, `aircraft`, `cars`

### Usage

```bash
uv run python src/linear_probe.py \
  --checkpoint_path data/checkpoints/LeJEPA_imagenet-1k/LV6_MV0_BS512_e100/last.ckpt \
  --model_name vit_large_patch16_224 \
  --proj_dim 512 \
  --datasets dtd cifar10 cifar100 flowers102 food101 pets \
  --fractions 0.01 0.10 1.0 \
  --epochs 100 \
  --optim L7 \
  --seed 0 \
  --num_workers 8 \
  --extract_batch_size 256 \
  --batch_size 512 \
  --device cuda \
  --wandb_project fewshot-JEPA \
  --wandb_run_name my_run
```

Via SLURM:

```bash
sbatch scripts/fewshot.sh
```

### Optimizer Presets

| Preset | Optimizer | LR | WD | Schedule |
|---|---|---|---|---|
| `L7` | Adam | 1e-2 | 0 | None |
| `L9` | SGD (momentum 0.9) | 1e-2 | 1e-6 | Cosine |

### Checkpoint Formats

Both Lightning `.ckpt` checkpoints (from `src/run_training_loop.py`) and legacy `.pth` checkpoints are supported. The backbone is extracted automatically; the projection head is discarded.

---

## Key Hyperparameters

| Parameter | Description | Paper default |
|---|---|---|
| `+reg` | Loss method: `LeJEPA` (MSE+SIGReg), `hybrid` (InfoNCE+SIGReg), `weighted_hybrid` | `LeJEPA` |
| `+model_name` | timm backbone identifier | `vit_large_patch16_224` |
| `+V_global` | Number of global views (224×224) | 2 |
| `+V_local` | Number of local views (96×96) | 6 |
| `+V_mixed` | Cross-instance views (same class, different image) | 0 |
| `+lamb` | SIGReg weight λ in `(1-λ)·pred + λ·SIGReg` | 0.05 |
| `+proj_dim` | Projection head output dimension | 512 |
| `+lr` | Peak learning rate (linear warmup 10 epochs + cosine decay) | 5e-4 |
| `+weight_decay` | AdamW weight decay (constant, no schedule) | 1e-2 |
| `+bs` | Total effective batch size across all GPUs | 512 |
| `+grad_accum` | Gradient accumulation steps | 1 |
| `+use_swa` | Stochastic Weight Averaging for target embeddings | False |

### View Configuration

| Config | `+V_global` | `+V_local` | `+V_mixed` | Dataset class |
|---|---|---|---|---|
| Standard (paper) | 2 | 6 | 0 | `HFDataset` |
| Mixed/cross-instance | 2 | 4 | 2 | `CrossInstanceDataset` |

---

## Directory Structure

```
jepa_tests/
├── src/
│   ├── run_training_loop.py   # Main entry point (Hydra)
│   ├── trainer.py             # Lightning modules (JEPATrainer, SimCLRTrainer, ...)
│   ├── encoder.py             # ViT backbone + MLP projection head
│   ├── ds.py                  # HFDataset, CrossInstanceDataset, collate_views
│   ├── linear_probe.py        # Few-shot transfer evaluation
│   ├── extract_features.py    # Offline feature extraction to disk
│   ├── stats.py               # Representation metrics (effective rank, LID, ...)
│   └── losses/
│       ├── loss.py            # LeJEPA, SIGReg, SimCLR, VICReg losses
│       └── lploss.py          # LpJEPA / RDMReg losses
├── scripts/
│   ├── train.sh               # Single-GPU SLURM job
│   ├── train_ddp.sh           # Multi-GPU DDP SLURM job
│   └── fewshot.sh             # Few-shot evaluation SLURM job
├── data/
│   ├── hub/                   # ImageNet-1K parquet shards
│   ├── cache/                 # inet100 parquet shards
│   └── checkpoints/           # Saved model checkpoints
├── logs/                      # SLURM stdout/stderr logs
├── pyproject.toml
└── uv.lock
```
