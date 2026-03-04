#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=80G
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:rtx_pro_6000:1

set -e

source ${env_activate}

echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version


export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ──────────────────────────────────────────────────────────────
# LeJEPA – Table 2 replication (ViT-L / IN-1K / 100 epochs)
#
# Model:       vit_large_patch16_224   (~304M params)
# Views:       V=8  (2 global 224×224 + 6 local 96×96)
# Loss:        (1-λ)·MSE + λ·SIGReg,  λ=0.05
# SIGReg:      Epps-Pulley, M=1024, 17 knots, domain [-5,5]
# Optimizer:   AdamW  (betas 0.9, 0.95)
# LR:          cross-validate {5e-3, 5e-4};  linear warmup + cosine anneal
# WD:          cross-validate {1e-1, 1e-2, 1e-5};  constant (no schedule)
# Batch size:  ≥128
# Architecture: No predictor, no register tokens, optional SWA
# ──────────────────────────────────────────────────────────────
srun python src/run_training_loop.py \
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
  +proj_dim=64 \
  +grad_accum=1 \
  +num_workers=16 \
  +prefetch_factor=2 \
  +device=cuda \
  +distributed=False \
  +world_size=1 \
  +seed=0 \
  +log_interval=40 \
  +use_swa=False

# ──────────────────────────────────────────────────────────────
# Uncomment below for ConvNeXtV2-H variant (~660M params).
# Same hyperparameter search space; SWA is less beneficial here.
# ──────────────────────────────────────────────────────────────
# srun python src/run_training_loop.py \
#   +reg=LeJEPA \
#   +model_name=convnextv2_huge \
#   +dataset=imagenet-1k \
#   +epochs=100 \
#   +bs=128 \
#   +lr=5e-4 \
#   +weight_decay=1e-2 \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=6 \
#   +V_mixed=0 \
#   +global_img_size=224 \
#   +local_img_size=96 \
#   +proj_dim=512 \
#   +grad_accum=2 \
#   +num_workers=8 \
#   +prefetch_factor=2 \
#   +device=cuda \
#   +distributed=False \
#   +world_size=1 \
#   +seed=0 \
#   +log_interval=40 \
#   +use_swa=False

