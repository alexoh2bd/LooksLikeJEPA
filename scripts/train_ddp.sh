#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=320:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4

# ===========================================================================
# Fail fast
# ===========================================================================
set -e

# ===========================================================================
# Environment
# ===========================================================================
source /home/users/aho13/jepa_tests/.venv/bin/activate

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
# HF dataset cache — node-local /tmp avoids NFS "stale file handle" errors
#
#   Each node gets its own cache. This is fine because HF datasets are
#   read-only after the first load. Both nodes will independently load
#   from the parquet shards on the shared filesystem into their local /tmp.
#   If you see load imbalance, switch to a shared NFS path instead:
#     export HF_DATASETS_CACHE=/path/on/shared/nfs/hf_cache
#     export HF_DATASETS_DISABLE_FILE_LOCKING=1
# ===========================================================================
export HF_DATASETS_CACHE=/tmp/hf_datasets_${SLURM_JOB_ID}
mkdir -p "$HF_DATASETS_CACHE"

# ===========================================================================
# Diagnostics (useful for debugging; safe to keep)
# ===========================================================================
echo "=== Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION ==="
echo "Nodes: $SLURM_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
which python
python --version

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
#   +world_size=8 \
#   +num_nodes=2 \
#   +seed=0 \
#   +log_interval=200 \
#   +use_swa=False

# ===========================================================================
# Training — PHN (uncomment to run instead of baseline)
#
#   Same hyperparameters as baseline, plus neighbor views.
#   Comment out the baseline srun above and uncomment this block.
# ===========================================================================
srun uv run src/run_training_loop.py \
  +reg=LeJEPA \
  +model_name=vit_large_patch14_224 \
  +dataset=imagenet-1k \
  +epochs=100 \
  +bs=512 \
  +lr=5e-4 \
  +weight_decay=5e-2 \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +V_mixed=0 \
  +global_img_size=224 \
  +local_img_size=98 \
  +proj_dim=512 \
  +grad_accum=1 \
  +num_workers=7 \
  +prefetch_factor=3 \
  +device=cuda \
  +distributed=True \
  +world_size=8 \
  +num_nodes=2 \
  +seed=0 \
  +log_interval=200 \
  +use_swa=False \
  +phn=True \
  +phn_neighbor_indices_path="data/b3/imagenet1k_qwen3_vl/ranks/neighbors.npy" \
  +phn_neighbor_scores_path="data/b3/imagenet1k_qwen3_vl/ranks/neighbor_scores.npy" \
  +phn_p=64 \
  +V_neighbor=2 \
  +phn_neighbor_sampling="uniform" \
  +phn_pos_only=False \
  +phn_neighbor_start_epoch=20