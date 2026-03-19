# B3-LeJEPA Batch Mining Pipeline

Offline preprocessing pipeline that constructs semantically-informed training
batches for LeJEPA / SIGReg pretraining on ImageNet-1K.  Inspired by the B3
paper ("Breaking the Batch Barrier of Contrastive Learning via Smart Batch
Mining", arXiv:2505.11293), adapted for single-image self-supervised training.

---

## How it works

```
Teacher encoder                    FAISS top-k              Goldilocks zone
(timm ViT-L)     →  embeddings.npy  →  neighbors.npy  →  adjacency_list.pkl
                                                                  ↓
                                                          METIS partitioning
                                                                  ↓
                                                          clusters.json
                                                                  ↓
                                                       LeJEPABatchSampler
```

Each training batch contains `batch_size // K` complete clusters of size `K`.
Samples within a cluster are mutual hard negatives: they are similar enough
to be informative (rank ≥ p) but not so similar as to be false negatives
(rank < p).

---

## Directory layout

```
src/pipeline/
├── configs/
│   └── default.yaml           ← all hyperparameters
├── scripts/
│   ├── utils.py
│   ├── 01_extract_embeddings.py
│   ├── 02_build_rank_matrix.py
│   ├── 03_build_sparse_graph.py
│   ├── 04_cluster_metis.py
│   └── 05_export_batches.py
├── batch_sampler.py            ← importable LeJEPABatchSampler
├── __init__.py
└── run_pipeline.sh             ← sequential runner (no SLURM)

scripts/
└── data_preprocess.sh          ← SLURM job-chain launcher (recommended)
```

---

## Running on SLURM (recommended)

```bash
# From the project root — all paths are absolute inside the script
bash scripts/data_preprocess.sh
```

This submits 8 jobs with `--dependency=afterok` chaining:

| Step | SLURM job | Parallelism | GPU? |
|------|-----------|------------|------|
| Stage 1 extract | array (N_SHARDS tasks) | parquet shards split across tasks | ✅ |
| Stage 1 stitch  | 1 task | — | ❌ |
| Stage 2 FAISS   | array (N_SHARDS tasks) | row chunks of embeddings | ✅ |
| Stage 2 stitch  | 1 task | — | ❌ |
| Stage 3 directed edges | array (N_SHARDS tasks) | row slices of neighbors | ❌ |
| Stage 3 merge   | 1 task (160 GB RAM) | — | ❌ |
| Stage 4 METIS   | 1 task | — | ❌ |
| Stage 5 export  | 1 task | — | ❌ |

**Override any parameter via environment variables:**

```bash
export N_SHARDS=4          # default: 4 parallel tasks per array stage
export K=32                # cluster size
export CHECKPOINT=/path/to/lejepa.ckpt   # use a trained LeJEPA teacher
bash scripts/data_preprocess.sh
```

Monitor progress:

```bash
squeue -u $USER
# or watch a specific stage log:
tail -f logs/b3_s2_JOBID_TASKID.log
```

---

## Running locally / sequentially

```bash
cd /home/users/aho13/jepa_tests
bash src/pipeline/run_pipeline.sh
```

Or run each stage individually:

```bash
# Stage 1
uv run python src/pipeline/scripts/01_extract_embeddings.py \
    --model_name vit_large_patch16_224 \
    --output_dir outputs/imagenet1k_b3/embeddings

# Stage 2
uv run python src/pipeline/scripts/02_build_rank_matrix.py \
    --embeddings_path outputs/imagenet1k_b3/embeddings/embeddings.npy \
    --output_dir      outputs/imagenet1k_b3/ranks \
    --use_gpu_faiss

# Stage 3
uv run python src/pipeline/scripts/03_build_sparse_graph.py \
    --neighbors_path outputs/imagenet1k_b3/ranks/neighbors.npy \
    --output_dir     outputs/imagenet1k_b3/graph

# Stage 4
uv run python src/pipeline/scripts/04_cluster_metis.py \
    --graph_path outputs/imagenet1k_b3/graph/adjacency_list.pkl \
    --output_dir outputs/imagenet1k_b3/clusters --K 32

# Stage 5
uv run python src/pipeline/scripts/05_export_batches.py \
    --clusters_path outputs/imagenet1k_b3/clusters/clusters.json \
    --output_dir    outputs/imagenet1k_b3/batches
```

---

## Using the batch sampler in training

```python
from src.pipeline.batch_sampler import LeJEPABatchSampler
from torch.utils.data import DataLoader

sampler = LeJEPABatchSampler(
    clusters_path="outputs/imagenet1k_b3/clusters/clusters.json",
    batch_size=256,
    K=32,
    seed=42,
    epoch=0,
)
# Call sampler.set_epoch(epoch) at the start of each epoch for diversity.

loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)
```

### DDP / multi-GPU training

```python
sampler = LeJEPABatchSampler(
    clusters_path="...",
    batch_size=256,
    K=32,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
```

### Lightning integration

Override `train_dataloader` in your `LightningModule`:

```python
def train_dataloader(self):
    sampler = LeJEPABatchSampler(
        clusters_path=self.hparams.clusters_path,
        batch_size=self.hparams.batch_size,
        K=self.hparams.K,
        seed=self.hparams.seed,
        epoch=self.current_epoch,
    )
    return DataLoader(self.train_dataset, batch_sampler=sampler, num_workers=8)
```

---

## Key hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `p` | 30 | Skip top-p neighbors (false-negative filter) |
| `m` | 100 | Window size — use ranks [p, p+m) as hard negatives |
| `top_k` | 130 | p + m; how many neighbors FAISS retrieves per query |
| `K` | 32 | Samples per cluster (= samples per "hard negative group") |
| `batch_size` | 256 | Must be divisible by K; clusters_per_batch = batch_size // K |
| `N_SHARDS` | 4 | SLURM array width for Stages 1, 2, 3 |

---

## Memory requirements

| Stage | RAM needed | Notes |
|-------|-----------|-------|
| Stage 1 | ~8 GB GPU VRAM | ViT-L inference; CPU RAM < 16 GB |
| Stage 2 | ~8 GB GPU VRAM | FAISS index for 1.28M × 1024 embeddings = 5 GB |
| Stage 3 (per shard) | ~16 GB | Directed edge array: N/S × m × 8 bytes |
| Stage 3 merge | ~128 GB | Full 128M-edge int64 array + searchsorted |
| Stage 4 | ~32 GB | METIS adjacency list in RAM |
| Stage 5 | < 4 GB | JSON loading only |

> **Tip:** Reduce `m` (e.g. 50) or `N_SHARDS` if memory is constrained.
> The MEM_CPU variable in `data_preprocess.sh` controls the merge job memory.
