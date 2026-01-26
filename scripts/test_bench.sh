#!/bin/bash
#SBATCH --job-name=lejepa_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

# ./run_inf.sh python eval/mamf_bench.py

./run_inf.sh nsys profile \
  --trace=cuda,osrt,nvtx \
  --stats=true \
  --force-overwrite=true \
  --output=profile_output \
  python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=5 \
  +V_mixed=1 \
  +model_name=vit_base_patch16_224.dino \
  +save_prefix=vit_cross_instance \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +lr=5e-4 \
  +bs=256 \
  +epochs=100 \
  +num_workers=4 \
  +device=cuda \
  +prefetch_factor=4 \
  +view_selection=mixed \
  +benchmark=True