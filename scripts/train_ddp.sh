#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:a5000:4

'''
5
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:a6000:4

2
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=280:00:00
#SBATCH --gres=gpu:a6000:2


'''

# Fail fast
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate environment
source ${env_activate}

# Optional: debugging
echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version

# Run training with Lightning DDP (no manual srun needed)
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1


# Lightning handles DDP automatically when distributed=True and world_size>1

"""
# srun python src/run_training_loop.py \
#   +reg=hybrid \
#   +model_name=vit_large_patch16_224 \
#   +dataset=imagenet-1k \
#   +epochs=100 \
#   +bs=512 \
#   +lr=5e-4 \
#   +weight_decay=1e-2 \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=6 \
#   +V_mixed=0 \
#   +global_img_size=224 \
#   +local_img_size=96 \
#   +proj_dim=64 \
#   +grad_accum=1 \
#   +num_workers=7 \
#   +prefetch_factor=4 \
#   +device=cuda \
#   +distributed=True \
#   +world_size=4 \
#   +num_nodes=1 \
#   +seed=0 \
#   +log_interval=40 \
#   +use_swa=False
"""

# world size = 8
srun python src/run_training_loop.py \
  +reg=hybrid \
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
  +num_workers=5 \
  +prefetch_factor=2 \
  +device=cuda \
  +distributed=True \
  +world_size=8 \
  +num_nodes=2 \
  +seed=0 \
  +log_interval=400 \
  +use_swa=False

"""
# srun python src/run_training_loop.py \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=2 \
#   +V_mixed=2 \
#   +model_name=vit_base_patch16_224.dino \
#   +global_img_size=224 \
#   +local_img_size=96 \
#   +proj_dim=256 \
#   +lr=5e-4 \
#   +bs=512 \
#   +grad_accum=1 \
#   +epochs=100\
#   +num_workers=5 \
#   +device=cuda \
#   +prefetch_factor=2 \
#   +temperature=0.05 \
#   +dataset=inet100 \
#   +reg=LeJEPA \
#   +distributed=True \
#   +seed=0 \
#   +world_size=2 \
#   +log_interval=40 \
  
# world size = 4
# srun uv python eval/run_training_loop.py \
#   +lamb=0.05 \
#   +V_global=2 \
#   +V_local=8 \
#   +V_mixed=0 \
#   +model_name=vit_base_patch16_224.dino \
#   +global_img_size=224 \
#   +local_img_size=96 \
#   +proj_dim=256 \
#   +lr=5e-4 \
#   +bs=1024 \
#   +grad_accum=1 \
#   +epochs=100\
#   +num_workers=4 \
#   +device=cuda \
#   +prefetch_factor=2 \
#   +dataset=inet100 \
#   +reg=hybrid \
#   +distributed=True \
#   +seed=0 \
#   +world_size=4 \
#   +log_interval=40 \
"""