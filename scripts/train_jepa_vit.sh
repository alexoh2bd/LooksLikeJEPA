#!/bin/bash
#SBATCH --job-name=lejepa
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1


# Fail fast
set -e

# Activate environment
source /home/users/aho13/jepa_tests/env/bin/activate

# Optional: debugging
echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version

# Run training
export HYDRA_FULL_ERROR=1

# Memory-efficient training configuration:
# - Use smaller bs with accum_steps to simulate larger effective batch size
# - Example: bs=64 with accum_steps=4 gives effective batch size of 256
# - Gradient checkpointing is now enabled by default (saves 20-40GB)

./run_inf.sh python eval/run_JEPA.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=6 \
  +V_mixed=2 \
  +model_name=vit_base_patch16_224.dino \
  +save_prefix=vit_JEPA \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +lr=5e-4 \
  +bs=256 \
  +grad_accum=2\
  +epochs=100 \
  +num_workers=6 \
  +device=cuda \
  +prefetch_factor=4 \
  +benchmark=False \
  +dataset=inet100 \
  +reg=LeJEPA \