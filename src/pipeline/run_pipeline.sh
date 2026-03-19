#!/bin/bash
#SBATCH --job-name=data_preprocess
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx_pro_6000:1


# B3-LeJEPA Batch Mining Pipeline — End-to-end runner
#
# Run all 5 stages sequentially.  Each stage can also be run independently;
# see the individual script --help for all options.
#
# Usage:
#   cd /home/users/aho13/jepa_tests
#   bash src/pipeline/run_pipeline.sh
#
# Override defaults via environment variables, e.g.:
#   MODEL_NAME=vit_base_patch16_224 K=32 bash src/pipeline/run_pipeline.sh

set -euo pipefail

# ── Configurable variables (all can be overridden from the environment) ────
PIPELINE_DIR="src/pipeline"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"
CONFIG="${PIPELINE_DIR}/configs/default.yaml"

OUTPUT_BASE="${OUTPUT_BASE:-outputs/imagenet1k_b3}"
DATA_DIR="${DATA_DIR:-data/hub/datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data}"

MODEL_NAME="${MODEL_NAME:-vit_base_patch16_clip_224.openai}"
CHECKPOINT="${CHECKPOINT:-}"           # leave empty to use timm pretrained weights
BATCH_SIZE_INF="${BATCH_SIZE_INF:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE_IDS="${DEVICE_IDS:-0}"
USE_GPU_FAISS="${USE_GPU_FAISS:-true}"

TOP_K="${TOP_K:-130}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"

P="${P:-30}"
M="${M:-100}"

K="${K:-32}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
SEED="${SEED:-42}"

EXPORT_STATIC="${EXPORT_STATIC:-false}"    # set "true" to write per-epoch batch files

# ── Helper ─────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Stage 1: Extract embeddings ────────────────────────────────────────────
log "=== Stage 1: Extract teacher embeddings ==="
CKPT_FLAG=""
[[ -n "${CHECKPOINT}" ]] && CKPT_FLAG="--checkpoint_path ${CHECKPOINT}"

uv run python "${SCRIPTS_DIR}/01_extract_embeddings.py" \
    --model_name      "${MODEL_NAME}" \
    --data_dir        "${DATA_DIR}" \
    --output_dir      "${OUTPUT_BASE}/embeddings" \
    --batch_size      "${BATCH_SIZE_INF}" \
    --num_workers     "${NUM_WORKERS}" \
    --device_ids      "${DEVICE_IDS}" \
    --seed            "${SEED}" \
    ${CKPT_FLAG}

log "Stage 1 complete."

# ── Stage 2: Build rank matrix ──────────────────────────────────────────────
log "=== Stage 2: Build sparse rank matrix ==="
GPU_FLAG=""
[[ "${USE_GPU_FAISS}" == "true" ]] && GPU_FLAG="--use_gpu_faiss"

uv run python "${SCRIPTS_DIR}/02_build_rank_matrix.py" \
    --embeddings_path "${OUTPUT_BASE}/embeddings/embeddings.npy" \
    --output_dir      "${OUTPUT_BASE}/ranks" \
    --top_k           "${TOP_K}" \
    --chunk_size      "${CHUNK_SIZE}" \
    ${GPU_FLAG}

log "Stage 2 complete."

# ── Stage 3: Build sparse graph ─────────────────────────────────────────────
log "=== Stage 3: Build sparse adjacency graph ==="
uv run python "${SCRIPTS_DIR}/03_build_sparse_graph.py" \
    --neighbors_path  "${OUTPUT_BASE}/ranks/neighbors.npy" \
    --output_dir      "${OUTPUT_BASE}/graph" \
    --p               "${P}" \
    --m               "${M}"

log "Stage 3 complete."

# ── Stage 4: METIS clustering ────────────────────────────────────────────────
log "=== Stage 4: METIS community detection ==="
uv run python "${SCRIPTS_DIR}/04_cluster_metis.py" \
    --graph_path      "${OUTPUT_BASE}/graph/adjacency_list.pkl" \
    --output_dir      "${OUTPUT_BASE}/clusters" \
    --K               "${K}" \
    --seed            "${SEED}"

log "Stage 4 complete."

# ── Stage 5: Export batches ──────────────────────────────────────────────────
log "=== Stage 5: Export batch schedule ==="
STATIC_FLAG=""
[[ "${EXPORT_STATIC}" == "true" ]] && STATIC_FLAG="--export_static"

uv run python "${SCRIPTS_DIR}/05_export_batches.py" \
    --clusters_path   "${OUTPUT_BASE}/clusters/clusters.json" \
    --output_dir      "${OUTPUT_BASE}/batches" \
    --batch_size      "${BATCH_SIZE}" \
    --K               "${K}" \
    --num_epochs      "${NUM_EPOCHS}" \
    --seed            "${SEED}" \
    ${STATIC_FLAG}

log "Stage 5 complete."

# ── Summary ──────────────────────────────────────────────────────────────────
log "=== Pipeline complete ==="
log ""
log "Outputs:"
log "  Embeddings : ${OUTPUT_BASE}/embeddings/embeddings.npy"
log "  Neighbors  : ${OUTPUT_BASE}/ranks/neighbors.npy"
log "  Graph      : ${OUTPUT_BASE}/graph/adjacency_list.pkl"
log "  Clusters   : ${OUTPUT_BASE}/clusters/clusters.json"
log "  Sampler    : ${OUTPUT_BASE}/batches/batch_sampler.py"
log ""
log "Use the sampler in training:"
log "  from src.pipeline.batch_sampler import LeJEPABatchSampler"
log "  sampler = LeJEPABatchSampler('${OUTPUT_BASE}/clusters/clusters.json', batch_size=${BATCH_SIZE}, K=${K})"
