#!/bin/bash
#SBATCH --job-name=ddp_i1k
#SBATCH --output=log/%x/%j.log
#SBATCH --error=log/%x/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
# ViT-L + ImageNet + DataLoader can exceed 64G RSS (sacct MaxRSS ~50–70G observed) — OOM kills look like FAILED 135 / step CANCELLED.
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:2

# ===========================================================================
# Fail fast
# ===========================================================================
set -e

# Project root: sbatch spools this script; training uses relative src/...
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  cd "$SCRIPT_DIR/.." || exit 1
fi

# Do not use .venv/bin/python directly: it is often a symlink to ~/.local/share/uv/python/...
# which may be missing on compute nodes or after uv cache cleanup ("No such file or directory").
# `uv run python` resolves/fetches an interpreter from pyproject.toml (same as scripts/train.sh).
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not in PATH (expected e.g. ${HOME}/.local/bin/uv)" >&2
  exit 1
fi

# Vendored submodule is not a published wheel — expose package `stable_pretraining` on PYTHONPATH.
export PYTHONPATH="${PWD}/stable-pretraining${PYTHONPATH:+:${PYTHONPATH}}"

# ===========================================================================
# Environment
# ===========================================================================

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1

# ===========================================================================
# NCCL — single, consistent block for 2-node TCP fallback
#
#   NCCL_IB_DISABLE=1      : skip InfiniBand (unreliable on this cluster)
#   NCCL_P2P_DISABLE=1     : skip GPU-direct P2P (only works intra-node)
#   NCCL_SOCKET_IFNAME      : use only real ethernet; exclude docker/loopback
#                              Change to "eth0" or "eno1" if ^docker0,lo
#                              doesn't work — run `ip link` on both nodes
#                              to find the correct interface name.
#   NCCL_TIMEOUT            : 1 hour (generous; original 30 min was too tight)
#   NCCL_DEBUG              : set to INFO for first run to verify connectivity,
#                              then switch to WARN to reduce log noise.
# ===========================================================================
# --- OPTION A: InfiniBand (fast, try first) ---
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5                   # common Mellanox HCA; run `ibstat` to confirm
export NCCL_IB_GID_INDEX=3                # RoCEv2 default; try 0 if 3 fails
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5               # enable GPU-Direct RDMA if available
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO                     # INFO for first run to verify IB; switch to WARN after



# --- OPTION B: TCP fallback (reliable, slower) ---
# Uncomment these and comment out Option A if InfiniBand hangs.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_TIMEOUT=3600
# export NCCL_DEBUG=WARN

# ===========================================================================
# HF dataset cache — node-local scratch avoids NFS "stale file handle" errors
#
#   Prefer SLURM_TMPDIR (per-node job scratch) when the scheduler sets it; else /tmp.
#   Each node has its own directory. Ranks on a node share one cache dir; Python
#   serializes prep on local rank 0 then barrier (see prepare_hf_dataset_cache in ds.py).
#   Shared NFS cache (if you must): set HF_DATASETS_CACHE to the NFS path and add
#     export HF_DATASETS_DISABLE_FILE_LOCKING=1
# ===========================================================================
: "${SLURM_TMPDIR:=/tmp}"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets_${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"

# ===========================================================================
# Diagnostics (useful for debugging; safe to keep)
# ===========================================================================
echo "=== Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION ==="
echo "Nodes: $SLURM_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
uv run python -c "import stable_pretraining; print('stable_pretraining OK')"
uv run python --version


srun uv run python src/run_training_loop.py \
  +reg=LeJEPA \
  +model_name=vit_large_patch14_224 \
  +dataset=imagenet-1k \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=5e-2 \
  +lamb=0.02 \
  +V_global=2 \
  +V_local=5\
  +V_mixed=1 \
  +global_img_size=224 \
  +local_img_size=98 \
  +proj_dim=512 \
  +grad_accum=1 \
  +num_workers=6 \
  +prefetch_factor=2 \
  +device=cuda \
  +distributed=True \
  +world_size=2 \
  +num_nodes=1 \
  +seed=0 \
  +log_interval=50 \
  +use_swa=False \
  +torch_compile=true \
  +ckpt_every_n_epochs=2 \
  +sigreg_impl=author

# ===========================================================================
# Training — LeJEPA baseline (no PHN)
#
#   Matches LeJEPA repo "GOTO hyperparameters":
#     - vit_large_patch14_224  (paper uses ViT-L/14)
#     - weight_decay=5e-2      (repo default for ViTs)
#     - local_img_size=98      (repo default)
#     - proj_dim=64            (Table 1d best)
#     - 1024 slices, [-5,5], 17 integration points (your SIGReg defaults)
# ===========================================================================
# srun uv run python src/run_training_loop.py \
#   +reg=LeJEPA \
#   +model_name=vit_large_patch14_224 \
#   +dataset=imagenet-1k \
#   +epochs=100 \
#   +bs=512 \
#   +lr=5e-4 \
#   +weight_decay=5e-2 \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=6 \
#   +V_mixed=0 \
#   +global_img_size=224 \
#   +local_img_size=98 \
#   +proj_dim=512 \
#   +grad_accum=1 \
#   +num_workers=7 \
#   +prefetch_factor=2 \
#   +device=cuda \
#   +distributed=True \
#   +world_size=2\
#   +num_nodes=1\
#   +seed=0 \
#   +log_interval=50 \
#   +use_swa=False \
#   +torch_compile=true \
#   +ckpt_every_n_epochs=2 \
#   +sigreg_impl=legacy

# ===========================================================================
# Training — PHN (uncomment to run instead of baseline)
#
#   Same hyperparameters as baseline, plus neighbor views.
#   Comment out the baseline srun above and uncomment this block.
# ===========================================================================
# srun uv run src/run_training_loop.py \
#   +reg=LeJEPA \
#   +model_name=vit_large_patch14_224 \
#   +dataset=imagenet-1k \
#   +epochs=100 \
#   +bs=512 \
#   +lr=5e-4 \
#   +weight_decay=5e-2 \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=5 \
#   +V_mixed=0 \
#   +global_img_size=224 \
#   +local_img_size=98 \
#   +proj_dim=512 \
#   +grad_accum=1 \
#   +num_workers=7 \
#   +prefetch_factor=3 \
#   +device=cuda \
#   +distributed=True \
#   +world_size=8 \
#   +num_nodes=2 \
#   +seed=0 \
#   +log_interval=200 \
#   +use_swa=False \
#   +phn=True \
#   +phn_neighbor_indices_path="data/b3/imagenet1k_qwen3_vl/ranks/neighbors.npy" \
#   +phn_neighbor_scores_path="data/b3/imagenet1k_qwen3_vl/ranks/neighbor_scores.npy" \
#   +phn_p=64 \
#   +V_neighbor=1 \
#   +phn_neighbor_sampling="uniform" \
#   +phn_pos_only=False \
#   +phn_neighbor_same_label_only=False \
#   +ckpt_every_n_epochs=2
#   # Warmup: 6 self-only locals (auto = V_local+V_neighbor), then 4 self + 2 neighbor.
#   # Override with +phn_warmup_V_local=6 if needed.
#   # phn_neighbor_same_label_only: neighbor pool = top-phn_p teacher ranks ∩ same class as anchor.
#   #   +phn_neighbor_start_epoch=20 \
