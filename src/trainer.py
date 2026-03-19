"""
Unified Training Pipeline for JEPA and Contrastive Learning.

Supports gradient accumulation and is designed to be extended for DDP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import os

import numpy as np
import tqdm
import wandb
from losses.loss import simclr_loss, LeJEPA, SIGReg, weighted_hybrid
from losses.lploss import rectified_lp_jepa_loss, rdmreg_loss, choose_sigma_for_unit_var
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L
from stats import RepresentationMetrics
from save import save_checkpoint
from ds import HFDataset, CrossInstanceDataset, collate_views
from neighbor_index import NeighborIndex
from mixed_view_ds import MixedViewDataset, collate_mixed_views
from pipeline.batch_sampler import PosBatchSampler
from losses.misc import gather

logging.basicConfig(level=logging.INFO)


def _is_phn(cfg) -> bool:
    """True if PHN (neighbor-view) mode is enabled."""
    return getattr(cfg, "phn_neighbor_indices_path", None) is not None


def _make_worker_init_fn(base_seed: int):
    """Return worker_init_fn that seeds each DataLoader worker for reproducibility."""

    def init_fn(worker_id: int) -> None:
        import random
        import numpy as np
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return init_fn


@dataclass
class TrainerConfig:
    """Configuration for the LeJEPA pretraining pipeline.

    Hyper-parameter ranges for cross-validation (Section 6.1):
        LR  : {5e-3, 5e-4}   – linear warm-up + cosine annealing
        WD  : {1e-1, 1e-2, 1e-5} – held *constant* (no schedule)
        BS  : >= 128 (stable even at 128)
    """
    # Model  (paper: ViT-L ~304M or ConvNeXtV2-H ~660M)
    model_name: str = "vit_large_patch16_224"
    proj_dim: int = 512
    
    # Training
    bs: int = 256
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-2           
    grad_accum: int = 1
    max_grad_norm: float = 3.0
    seed: int = 0
    reg: str = "LeJEPA"
    
    # Data  (paper: ImageNet-1K, 100 epochs)
    dataset: str = "imagenet-1k"
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Views  (paper: V=8 total — 2 global 224×224 + 6 local 98×98)
    V_global: int = 2
    V_local: int = 6
    V_mixed: int = 0
    global_img_size: int = 224
    local_img_size: int = 98
    
    # Device
    device: str = "cuda"
    
    # Logging
    log_interval: int = 50
    save_interval: int = 50  # epochs

    # SWA (optional, small boost for ViT – Section 6.1)
    use_swa: bool = False

    # Reproducibility (slower; use for debugging or exact replication)
    reproducible: bool = False
    
    # DDP placeholders (to be set by distributed setup)
    rank: int = 0
    local_rank: int = 0
    distributed: bool = False
    world_size: int = 1

    @classmethod
    def from_hydra(cls, cfg) -> "TrainerConfig":
        """Create config from Hydra DictConfig."""
        return cls(
            model_name=cfg.get("model_name", "vit_large_patch16_224"),
            proj_dim=cfg.get("proj_dim", 512),
            bs=cfg.get("bs", 256),
            epochs=cfg.get("epochs", 100),
            lr=cfg.get("lr", 5e-4),
            weight_decay=cfg.get("weight_decay", 1e-2),
            grad_accum=cfg.get("grad_accum", 1),
            max_grad_norm=cfg.get("max_grad_norm", 3.0),
            dataset=cfg.get("dataset", "imagenet-1k"),
            num_workers=cfg.get("num_workers", 4),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            V_global=cfg.get("V_global", 2),
            V_local=cfg.get("V_local", 6),
            V_mixed=cfg.get("V_mixed", 0),
            global_img_size=cfg.get("global_img_size", 224),
            local_img_size=cfg.get("local_img_size", 98),
            device=cfg.get("device", "cuda"),
            log_interval=cfg.get("log_interval", 50),
            save_interval=cfg.get("save_interval", 50),
            reg=cfg.get("reg", "LeJEPA"),
            distributed=cfg.get("distributed", False),
            world_size=cfg.get("world_size", 1),
            seed=cfg.get("seed", 0),
            use_swa=cfg.get("use_swa", False),
            reproducible=cfg.get("reproducible", False),
        )


class BaseTrainer(L.LightningModule):
    """
    Base trainer class for SSL methods (JEPA, SimCLR, etc.)
    
    Lightning-compatible implementation.
    
    Subclasses must implement:
        - compute_loss(): Returns loss dict with 'total' key
        - get_method_name(): Returns string name of method
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        config: TrainerConfig,
        hydra_cfg: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'hydra_cfg'])
        
        self.config = config
        self.hydra_cfg = hydra_cfg
        
        # Models (no .to(device) - Lightning handles this)
        self.encoder = encoder
        self.probe = self._build_probe()
        self.encoder = torch.compile(
            self.encoder,
            mode="default",
            fullgraph=False,
        )
        self.probe = torch.compile(
            self.probe,
            mode="default",
            fullgraph=False,
        )

        # Optional SWA encoder for producing stable target embeddings (z̄).
        # Only the averaged parameters are used; no gradients flow through it.
        self.use_swa = config.use_swa
        if self.use_swa:
            self.swa_encoder = AveragedModel(encoder)
        
        # Augmentations (no .to(device) - Lightning moves these automatically)
        self.gpu_aug_global = self._build_gpu_aug_global()
        self.gpu_aug_local = self._build_gpu_aug_local()
        
        # Metrics
        self.repmetrics = RepresentationMetrics()

        # State
        self.best_acc = 0.0
        
        # Datasets (will be created in setup())
        self.train_ds = None
        self.test_ds = None
        self._phn_dataset = None
        self._phn_sampler = None
            
    @abstractmethod
    def compute_loss(
        self,
        global_views: List[torch.Tensor],
        local_views: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the SSL loss.
        
        Args:
            global_views: List of global view tensors, each (N, C, H, W)
            local_views: List of local view tensors, each (N, C, H, W)
            labels: Ground truth labels (N,)
            
        Returns:
            Dict with at least 'total' key for total loss, plus any method-specific losses
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the SSL method (e.g., 'LeJEPA', 'SimCLR')."""
        pass
    
    # ========== Lightning Hooks ==========
    
    def setup(self, stage: str):
        """Lightning hook called at the beginning of fit/test.

        Priority for training dataset:
          1. PHN (neighbor views) → MixedViewDataset
          2. Mixed/cross-instance  → CrossInstanceDataset
          3. Default               → HFDataset
        """
        if stage != "fit":
            return
        cfg = self.config
        test_split = "test" if cfg.dataset == "cifar10" else "val"

        if _is_phn(cfg):
            neighbor_index = NeighborIndex(
                indices_path=cfg.phn_neighbor_indices_path,
                scores_path=cfg.phn_neighbor_scores_path,
            )
            logging.info("NeighborIndex loaded: %s", neighbor_index)
            logging.info(
                "[DEBUG] Neighbor index path: %s",
                cfg.phn_neighbor_indices_path,
            )
            nbr_idx, nbr_sim = neighbor_index.get_positives(0, p=5)
            logging.info(
                "[DEBUG] First 5 neighbors of image 0: indices=%s  scores=%s",
                nbr_idx.tolist(),
                [f"{s:.4f}" for s in nbr_sim.tolist()],
            )
            self.train_ds = MixedViewDataset(
                split="train",
                neighbor_index=neighbor_index,
                V_global=cfg.V_global,
                V_self=cfg.V_local,
                V_neighbor=getattr(cfg, "V_neighbor", 2),
                p=getattr(cfg, "phn_p", 64),
                min_similarity=getattr(cfg, "phn_min_similarity", 0.0),
                neighbor_sampling=getattr(cfg, "phn_neighbor_sampling", "uniform"),
                neighbor_start_epoch=getattr(cfg, "phn_neighbor_start_epoch", 0),
                global_img_size=cfg.global_img_size,
                local_img_size=cfg.local_img_size,
                dataset=cfg.dataset,
                seed=cfg.seed,
            )
            self._phn_dataset = self.train_ds
            # Cross-check: do image 0's neighbors have the same label?
            labels = np.array(self.train_ds.ds["label"], dtype=np.int64)
            nbr_idx, _ = neighbor_index.get_positives(0, p=5)
            if len(nbr_idx) > 0:
                query_label = labels[0]
                nbr_labels = labels[nbr_idx]
                same_count = (nbr_labels == query_label).sum()
                logging.info(
                    "[DEBUG] Image 0 label=%d, neighbor labels=%s, same-label count=%d/5",
                    int(query_label),
                    nbr_labels.tolist(),
                    int(same_count),
                )
        elif cfg.V_mixed > 0:
            self.train_ds = CrossInstanceDataset(
                "train",
                V_global=cfg.V_global,
                V_local=cfg.V_local,
                V_mixed=cfg.V_mixed,
                local_img_size=cfg.local_img_size,
                global_img_size=cfg.global_img_size,
                dataset=cfg.dataset,
                seed=cfg.seed,
            )
        else:
            self.train_ds = HFDataset(
                "train",
                V_global=cfg.V_global,
                V_local=cfg.V_local,
                local_img_size=cfg.local_img_size,
                global_img_size=cfg.global_img_size,
                dataset=cfg.dataset,
                seed=cfg.seed,
            )

        self.test_ds = HFDataset(
            test_split,
            V_global=1,
            V_local=0,
            local_img_size=cfg.local_img_size,
            global_img_size=cfg.global_img_size,
            dataset=cfg.dataset,
            seed=cfg.seed,
        )
    
    def _build_probe(self) -> nn.Module:
        """Build the linear probe for evaluation."""
        feat_dim = self.encoder.feat_dim
        ds = self.config.dataset
        num_classes = {
            "imagenet-1k": 1000,
            "inet100": 100,
            "cifar10": 10,
        }.get(ds, 1000)
        
        probe = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )
        return probe
    
    @property
    def per_device_batch_size(self) -> int:
        """Compute per-device batch size for DDP.
        
        With DDP, each GPU runs independently with its own dataloader.
        To achieve an effective batch size of `bs`, each GPU should process `bs // world_size`.
        """
        world_size = self.config.world_size if self.config.distributed else 1
        per_device_bs = self.config.bs // world_size
        if per_device_bs == 0:
            raise ValueError(f"Batch size {self.config.bs} is too small for {world_size} GPUs")
        return per_device_bs
    
    def train_dataloader(self):
        """Lightning hook for training dataloader.

        PHN + phn_pos_only → PosBatchSampler (cluster batches) + collate_mixed_views
        PHN (no sampler)   → shuffle DataLoader              + collate_mixed_views
        Default            → shuffle DataLoader              + collate_views
        """
        if self.train_ds is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        cfg = self.config

        if _is_phn(cfg) and getattr(cfg, "phn_pos_only", True):
            rank = self.trainer.global_rank if self.trainer is not None else 0
            world_size = self.trainer.world_size if self.trainer is not None else 1
            sampler = PosBatchSampler(
                dataset_size=len(self.train_ds),
                batch_size=self.per_device_batch_size,
                seed=cfg.seed,
                epoch=0,
                drop_last=True,
                rank=rank,
                world_size=world_size,
            )
            self._phn_sampler = sampler
            kw = dict(
                batch_sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=cfg.num_workers > 0,
                prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                collate_fn=collate_mixed_views,
            )
            if getattr(cfg, "reproducible", False) and cfg.num_workers > 0:
                kw["worker_init_fn"] = _make_worker_init_fn(cfg.seed)
            return DataLoader(self.train_ds, **kw)

        if _is_phn(cfg):
            self._phn_sampler = None
            g = torch.Generator()
            g.manual_seed(cfg.seed)
            kw = dict(
                batch_size=self.per_device_batch_size,
                shuffle=not cfg.distributed,
                drop_last=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                generator=g,
                persistent_workers=cfg.num_workers > 0,
                prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                collate_fn=collate_mixed_views,
            )
            if getattr(cfg, "reproducible", False) and cfg.num_workers > 0:
                kw["worker_init_fn"] = _make_worker_init_fn(cfg.seed)
            return DataLoader(self.train_ds, **kw)

        g = torch.Generator()
        g.manual_seed(cfg.seed)
        kw = dict(
            batch_size=self.per_device_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            generator=g,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            collate_fn=collate_views,
        )
        if getattr(cfg, "reproducible", False) and cfg.num_workers > 0:
            kw["worker_init_fn"] = _make_worker_init_fn(cfg.seed)
        return DataLoader(self.train_ds, **kw)

    def val_dataloader(self):
        """Lightning hook for validation dataloader.
        PHN + DDP uses DistributedSampler for val parity with non-PHN."""
        if self.test_ds is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        cfg = self.config
        sampler = None
        if _is_phn(cfg) and cfg.distributed:
            sampler = DistributedSampler(self.test_ds, shuffle=False)
        kw = dict(
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=cfg.distributed,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )
        if getattr(cfg, "reproducible", False) and cfg.num_workers > 0:
            kw["worker_init_fn"] = _make_worker_init_fn(cfg.seed)
        return DataLoader(self.test_ds, **kw)

    def on_train_epoch_start(self):
        """Advance per-epoch seeds for PHN neighbor sampling and PosBatchSampler."""
        if self._phn_sampler is not None:
            self._phn_sampler.set_epoch(self.current_epoch)
        if self._phn_dataset is not None:
            self._phn_dataset.set_epoch(self.current_epoch)
            start_epoch = getattr(self.config, "phn_neighbor_start_epoch", 0)
            if start_epoch > 0 and self.current_epoch == start_epoch:
                logging.info(
                    "PHN curriculum: introducing neighbor views at epoch %d",
                    self.current_epoch,
                )
                # Save model weights at transition (pure self-view checkpoint)
                if self.trainer.is_global_zero:
                    from lightning.pytorch.callbacks import ModelCheckpoint
                    ckpt_dir = None
                    for cb in self.trainer.callbacks:
                        if isinstance(cb, ModelCheckpoint) and getattr(cb, "dirpath", None):
                            ckpt_dir = cb.dirpath
                            break
                    if ckpt_dir is not None:
                        path = os.path.join(ckpt_dir, f"epoch_{start_epoch}_pre_phn.ckpt")
                        self.trainer.save_checkpoint(path)
                        logging.info("Saved pre-PHN checkpoint to %s", path)
    
    def _build_gpu_aug_global(self) -> v2.Compose:
        """Build GPU augmentation pipeline for global views."""
        return v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _build_gpu_aug_local(self) -> v2.Compose:
        """Build GPU augmentation pipeline for local views."""
        return v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),

            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def configure_optimizers(self):
        """AdamW with linear warm-up + cosine annealing on LR only.

        Weight decay is held constant (no schedule) per paper Section 6.1.
        5e-2 for ViT, 5e-4 for ResNet/ConvNeXt. Final LR = lr/1000.
        """
        model_name = self.config.model_name.lower()
        wd = 5e-2 if model_name.startswith("vit") else 5e-4
        optimizer = torch.optim.AdamW([
            {
                'params': self.encoder.parameters(),
                'lr': self.config.lr,
                'weight_decay': wd,
                'betas': (0.9, 0.95),
            },
            {
                'params': self.probe.parameters(),
                'lr': 3e-3,
                'weight_decay': 0.0,
                'betas': (0.9, 0.95),
            }
        ])
        
        steps_per_epoch = (
            len(self.train_ds) // self.config.bs // self.config.grad_accum
        )
        warmup_steps = steps_per_epoch * 10  # 10 epoch warmup
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_steps
        )
        eta_min = self.config.lr / 1000
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps, 
            eta_min=eta_min
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Lightning hook for training step."""
        vs, y = batch
        cfg = self.config

        # Separate views by resolution
        global_views = [vs[i] for i in range(len(vs)) if vs[i].shape[-1] == cfg.global_img_size]
        local_views = [vs[i] for i in range(len(vs)) if vs[i].shape[-1] == cfg.local_img_size]
        
        # Apply GPU augmentations
        global_views = [self.gpu_aug_global(g) for g in global_views]
        local_views = [self.gpu_aug_local(l) for l in local_views]
        
        # Compute SSL loss
        loss_dict = self.compute_loss(global_views, local_views, y)
        
        # Log method-specific metrics
        for k, v in loss_dict.items():
            if k != "total":
                self.log(f"train/{k}", v, on_step=True, on_epoch=False, sync_dist=True)
        
        # Lightning's accumulate_grad_batches already handles gradient averaging properly
        # Do NOT divide by grad_accum here - that would double-scale the loss
        return loss_dict["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        """Lightning hook for validation step."""
        vs, y = batch
        
        emb, _ = self.encoder(vs)
        emb_flat = emb.flatten(0, 1)
        logits = self.probe(emb_flat)
        
        acc = (logits.argmax(1) == y).float().mean()
        
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"emb": emb_flat, "labels": y}
    
    def on_validation_epoch_end(self):
        """Compute additional metrics at end of validation."""
        pass

    def on_train_epoch_end(self):
        """Update SWA model at the end of each training epoch."""
        if self.use_swa:
            self.swa_encoder.update_parameters(self.encoder)
    
    def on_save_checkpoint(self, checkpoint):
        """Lightning hook for saving additional state."""
        checkpoint['best_acc'] = self.best_acc
        checkpoint['method_name'] = self.get_method_name()


# ========== Concrete Trainer Implementations ==========

class JEPATrainer(BaseTrainer):
    """Trainer for LeJEPA / SIGReg based methods.

    Architecture rules (paper Section 6.1):
      - No explicit predictor network — representations compared directly.
      - No register tokens (handled in Encoder).
      - Optional SWA on the encoder for target embeddings z̄ (handled in BaseTrainer).
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        config: TrainerConfig,
        lamb: float = 0.05,
        w: float = 0.5,
        hydra_cfg: Optional[Any] = None,
    ):
        super().__init__(encoder, config, hydra_cfg)
        self.sigreg = SIGReg()
        self.lamb = lamb
        self.w = w

    def get_method_name(self) -> str:
        return self.config.reg
    
    def compute_loss(
        self,
        global_views: List[torch.Tensor],
        local_views: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
    
        # Forward through encoder
        all_views = global_views + local_views
        all_emb, all_proj = self.encoder(all_views)
        
        if self.config.distributed:
            gathered_proj = self.all_gather(all_proj, sync_grads=True)
            gathered_emb = self.all_gather(all_emb, sync_grads=True)
            all_proj = gathered_proj.flatten(0, 1)
            all_emb = gathered_emb.flatten(0, 1)
            labels = self.all_gather(labels, sync_grads=False).flatten(0, 1)

        global_proj = all_proj[:, :self.config.V_global, :]

        # Optionally compute SWA target for more stable global centre
        swa_target = None
        if self.use_swa:
            with torch.no_grad():
                _, swa_proj = self.swa_encoder(global_views)
                swa_target = swa_proj.mean(dim=1, keepdim=True)  # (N, 1, D)

        if self.config.reg == "weighted_hybrid":
            ssl_loss, lejepa_loss, cl_loss, sigreg_loss = weighted_hybrid(
                global_proj, all_proj, self.sigreg, w=self.w, lamb=self.lamb
            )
            pred_loss = lejepa_loss
        else:
            ssl_loss, pred_loss, sigreg_loss = LeJEPA(
                all_proj, self.config.V_global, self.sigreg, self.lamb, reg=self.config.reg,
                target=swa_target,
                global_step=self.global_step,
            )
            cl_loss = torch.tensor(0.0, device=all_proj.device)

        # Compute probe loss
        if getattr(self.config, "V_neighbor", 0) > 0:
            V = len(global_views) + len(local_views)
        else:
            V = self.config.V_global + self.config.V_local + self.config.V_mixed
        y_rep = labels.repeat_interleave(V)
        yhat = self.probe(all_emb.flatten(0, 1).detach())
        probe_loss = F.cross_entropy(yhat, y_rep)

        total_loss = ssl_loss + probe_loss
        return {
            "total_loss": total_loss,
            "sigreg_loss": sigreg_loss,
            "prediction_loss": pred_loss,
            "cl_loss": cl_loss,
            "probe_loss": probe_loss,
        }


class LpJEPATrainer(BaseTrainer):
    """Trainer for Rectified LpJEPA / RDMReg based methods.

    Uses sliced Wasserstein distance (RDMReg) as the regularizer instead of SIGReg.
    Supports hybrid mode (lp_hybrid) which replaces MSE invariance with InfoNCE.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: TrainerConfig,
        invariance_weight: float = 25.0,
        rdm_reg_weight: float = 25.0,
        lp_norm_parameter: float = 1.0,
        mean_shift_value: float = 0.0,
        target_distribution: str = "rectified_lp_distribution",
        num_projections: int = 256,
        hydra_cfg: Optional[Any] = None,
    ):
        super().__init__(encoder, config, hydra_cfg)
        self.invariance_weight = invariance_weight
        self.rdm_reg_weight = rdm_reg_weight
        self.lp_norm_parameter = lp_norm_parameter
        self.mean_shift_value = mean_shift_value
        self.target_distribution = target_distribution
        self.num_projections = num_projections

        # Pre-compute sigma so Var(ReLU(X)) = 1 for the chosen p and mu.
        # This is expensive (mpmath bisection) but only runs once at init.
        if target_distribution == "rectified_lp_distribution":
            self.chosen_sigma = choose_sigma_for_unit_var(
                lp_norm_parameter, mean_shift_value
            )
        else:
            from losses.lploss import determine_sigma_for_lp_dist
            self.chosen_sigma = determine_sigma_for_lp_dist(lp_norm_parameter)

        logging.info(
            f"LpJEPATrainer: p={lp_norm_parameter}, mu={mean_shift_value}, "
            f"sigma={self.chosen_sigma:.6f}, target={target_distribution}"
        )

    def get_method_name(self) -> str:
        return self.config.reg

    def _random_projection_vectors(self, D: int, device, dtype) -> torch.Tensor:
        """Generate random orthonormal projection matrix (num_proj, D).
        Seeded with global_step for identical projections across DDP ranks."""
        g = torch.Generator(device=device)
        g.manual_seed(self.global_step)
        P = torch.randn(self.num_projections, D, device=device, dtype=dtype, generator=g)
        P = P / P.norm(p=2, dim=1, keepdim=True)
        return P

    def compute_loss(
        self,
        global_views: List[torch.Tensor],
        local_views: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        all_views = global_views + local_views
        all_emb, all_proj = self.encoder(all_views)

        # DDP gather: standard LpJEPA gathers only all_emb/labels (for probe);
        # RDMReg gather happens inside rectified_lp_jepa_loss via misc.gather.
        # lp_hybrid needs full batch for SimCLR, so we gather all_proj too.
        if self.config.distributed:
            gathered_emb = self.all_gather(all_emb, sync_grads=True)
            all_emb = gathered_emb.flatten(0, 1)
            labels = self.all_gather(labels, sync_grads=False).flatten(0, 1)
            all_proj = gather(all_proj)  # flatten only if (N,V,D) -> need (N*V, D) for downstream

        global_proj = all_proj[:, :self.config.V_global, :]
        local_proj = all_proj[:, self.config.V_global:, :]

        # z1 = global center, z2 = local  —  (N, 1, D) (N, Vl, D)
        z1 = global_proj.mean(dim=1, keepdim=True)
        z2 = local_proj

        proj_vecs = self._random_projection_vectors(
            z1.shape[-1], z1.device, z1.dtype
        )
        # Standard LpJEPA: MSE + RDMReg
        ssl_loss, sim_loss, reg_loss = rectified_lp_jepa_loss(
            z1, z2, proj_vecs, self.target_distribution,
            self.invariance_weight, self.rdm_reg_weight,
            self.mean_shift_value, self.lp_norm_parameter, self.chosen_sigma,
        )

        # Probe loss
        V = len(global_views) + len(local_views)
        y_rep = labels.repeat_interleave(V)
        yhat = self.probe(all_emb.flatten(0, 1).detach())
        probe_loss = F.cross_entropy(yhat, y_rep)

        total_loss = ssl_loss + probe_loss
        return {
            "total_loss": total_loss,
            "sim_loss": sim_loss,
            "rdm_reg_loss": reg_loss,
            "probe_loss": probe_loss,
        }


class SimCLRTrainer(BaseTrainer):
    """Trainer for SimCLR / contrastive learning methods."""
    
    def __init__(
        self,
        encoder: nn.Module,
        config: TrainerConfig,
        temperature: float = 0.5,
        hydra_cfg: Optional[Any] = None,
    ):
        super().__init__(encoder, config, hydra_cfg)
        self.temperature = temperature
    
    def get_method_name(self) -> str:
        return "SimCLR"
    
    def compute_loss(
        self,
        global_views: List[torch.Tensor], # N, Vg, 226, 226
        local_views: List[torch.Tensor], # N, Vl, 226, 226
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        all_views = global_views + local_views
        all_emb, all_proj = self.encoder(all_views)
        
        # Gather across GPUs if using DDP (Lightning auto-detects)
        if self.config.distributed:
            gathered_proj = self.all_gather(all_proj, sync_grads=True)
            gathered_labels = self.all_gather(labels, sync_grads=False)
            gathered_emb = self.all_gather(all_emb, sync_grads=True) 
            
            all_proj = gathered_proj.flatten(0, 1)
            all_emb = gathered_emb.flatten(0, 1)  
            labels = gathered_labels.flatten(0, 1)
            
        global_proj = all_proj[:, :self.config.V_global, :]
        local_proj = all_proj[:, self.config.V_global:, :]
        
        # SimCLR loss
        cl_loss = simclr_loss(global_proj, local_proj,  temperature=self.temperature)
        # Compute probe loss
        V = self.config.V_global + self.config.V_local+self.config.V_mixed
        y_rep = labels.repeat_interleave(V)
        yhat = self.probe(all_emb.flatten(0, 1).detach())
        probe_loss = F.cross_entropy(yhat, y_rep)
        total_loss = cl_loss + probe_loss
        

        return {
            "total_loss": total_loss,
            "cl_loss": cl_loss,
            "probe_loss": probe_loss,
        }
