#!/bin/bash
#SBATCH --job-name=fewshot
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a5000:1


# Fail fast
set -e

# Activate environment
source /home/users/aho13/jepa_tests/.venv/bin/activate

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
# Feature extraction: last-2-layer CLS concat + LayerNorm
# Optimizer: AdamW, wd=1e-6, linear warmup + cosine annealing
uv run python src/linear_probe.py \
    --checkpoint_path data/checkpoints/LeJEPA_imagenet-1k/LV4_MV2_BS512_e100_ddp4/last.ckpt \
    --model_name vit_large_patch14_224 \
    --proj_dim 512 \
    --datasets dtd cifar10 cifar100 flowers102 food101 pets cars \
    --k_shot 1 10 all \
    --seeds 0 1 2 \
    --lr 1e-2 \
    --num_workers 8 \
    --extract_batch_size 512 \
    --batch_size 512 \
    --device cuda \
    --wandb_project fewshot-JEPA \

# uv run python src/linear_probe.py \
#     --checkpoint_path data/checkpoints/LpJEPA_imagenet-1k/LV4_MV0_NV2_P64_BS512_e100_ddp5/last.ckpt \
#     --model_name vit_large_patch14_224 \
#     --proj_dim 512 \
#     --datasets dtd cifar10 cifar100 flowers102 food101 pets cars \
#     --k_shot 1 10 all \
#     --seeds 0 1 2 \
#     --lr 1e-2 \
#     --num_workers 8 \
#     --extract_batch_size 512 \
#     --batch_size 512 \
#     --device cuda \
#     --wandb_project fewshot-JEPA \
