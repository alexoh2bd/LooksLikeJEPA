#!/bin/bash
#SBATCH --job-name=fewshot
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a5000:1


# Fail fast
set -e

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

# Run training
python src/linear_probe.py \
    --checkpoint_path data/checkpoints/SimCLR_inet100_vit_base_patch16_224.dino/LV4\|MV0\|BS256_e100/V4/checkpoint_lastSimCLR_e99_inet100_4LV.pth \
    --model_name vit_base_patch16_224.dino \
    --proj_dim 256 \
    --datasets dtd cifar10 cifar100 flowers102 food101 pets \
    --fractions 0.01 0.10 1.0 \
    --epochs 1 \
    --optim L7 \
    --seed 0 \
    --num_workers 8 \
    --extract_batch_size 256 \
    --batch_size 512 \
    --device cuda \
    --wandb_project fewshot-JEPA \
    --wandb_run_name LpJEPA_hybrid_200