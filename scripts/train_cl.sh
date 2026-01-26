#!/bin/bash
#SBATCH --job-name=simclr_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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

./run_inf.sh python eval/run_CL.py \
  +model_name=vit_base_patch16_224.dino \
  +proj_dim=512 \
  +lr=5e-4 \
  +global_img_size=224 \
  +local_img_size=96 \
  +bs=256 \
  +grad_accum=1 \
  +epochs=100 \
  +weight_decay=0.0 \
  +num_workers=4\
  +device=cuda \
  +dataset=inet100 \
  +temperature=0.5 \
  +prefetch_factor=4\
  +V_global=2\
  +V_local=8\
  +V_mixed=0
