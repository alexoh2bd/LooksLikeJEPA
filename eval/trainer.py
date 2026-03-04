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
from abc import  abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import tqdm
import wandb
from loss import simclr_loss, LeJEPA, SIGReg
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L

from stats import RepresentationMetrics
from save import save_checkpoint
from ds import HFDataset, CrossInstanceDataset, collate_views

logging.basicConfig(level=logging.INFO)


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    # Model
    model_name: str = "vit_base_patch16_224.dino"
    proj_dim: int = 512
    
    # Training
    bs: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-2
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    seed: int = 0
    reg: str = "LeJEPA"
    
    # Data
    dataset: str = "inet100"
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Views
    V_global: int = 2
    V_local: int = 4
    V_mixed: int = 1
    global_img_size: int = 224
    local_img_size: int = 96
    
    # Device
    device: str = "cuda"
    
    # Logging
    log_interval: int = 50
    save_interval: int = 50  # epochs
    
    # DDP placeholders (to be set by distributed setup)
    rank: int = 0
    local_rank: int = 0
    distributed: bool = False
    world_size: int = 1

  



    @classmethod
    def from_hydra(cls, cfg) -> "TrainerConfig":
        """Create config from Hydra DictConfig."""
        return cls(
            model_name=cfg.get("model_name", "vit_base_patch16_224.dino"),
            proj_dim=cfg.get("proj_dim", 512),
            bs=cfg.get("bs", 256),
            epochs=cfg.get("epochs", 100),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 5e-2),
            grad_accum=cfg.get("grad_accum", 1),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            dataset=cfg.get("dataset", "inet100"),
            num_workers=cfg.get("num_workers", 4),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            V_global=cfg.get("V_global", 2),
            V_local=cfg.get("V_local", 4),
            V_mixed=cfg.get("V_mixed", 0),
            global_img_size=cfg.get("global_img_size", 224),
            local_img_size=cfg.get("local_img_size", 96),
            device=cfg.get("device", "cuda"),
            log_interval=cfg.get("log_interval", 50),
            save_interval=cfg.get("save_interval", 50),
            reg = cfg.get("reg", "LeJEPA"),
            distributed=cfg.get("distributed", False),
            world_size=cfg.get("world_size", 1),
            seed = cfg.get("seed", 0),
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
        # self.effective_bs = config.batch_size
        # self.real_bs =config.batch_size // (config.grad_accum * config.)
        
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
        """Lightning hook called at the beginning of fit/test."""
        if stage == "fit":
            cfg = self.config
            view_selection = "random" if cfg.V_mixed == 0 else "mixed"
            
            # Training dataset
            if view_selection == "mixed":
                self.train_ds = CrossInstanceDataset(
                    "train",
                    V_global=cfg.V_global,
                    V_local=cfg.V_local,
                    V_mixed=cfg.V_mixed,
                    local_img_size=cfg.local_img_size,
                    global_img_size=cfg.global_img_size,
                    dataset=cfg.dataset,
                )
            else:
                self.train_ds = HFDataset(
                    "train",
                    V_global=cfg.V_global,
                    V_local=cfg.V_local,
                    local_img_size=cfg.local_img_size,
                    global_img_size=cfg.global_img_size,
                    dataset=cfg.dataset,
                )
            
            # Test dataset
            test_split = "test" if cfg.dataset == "cifar10" else "val"
            self.test_ds = HFDataset(
                test_split,
                V_global=1,
                V_local=0,
                local_img_size=cfg.local_img_size,
                global_img_size=cfg.global_img_size,
                dataset=cfg.dataset,
            )
    
    def _build_probe(self) -> nn.Module:
        """Build the linear probe for evaluation."""
        feat_dim = self.encoder.feat_dim
        num_classes = 100 if self.config.dataset == "inet100" else 10
        
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
        """Lightning hook for training dataloader."""
        if self.train_ds is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        
        # Note: shuffle=False because Lightning's DistributedSampler handles shuffling
        # When use_distributed_sampler=True, Lightning wraps the dataloader automatically
        return DataLoader(
            self.train_ds,
            batch_size=self.per_device_batch_size,
            shuffle=not self.config.distributed,  # DistributedSampler handles shuffle in DDP
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            collate_fn=collate_views,
        )
    
    def val_dataloader(self):
        """Lightning hook for validation dataloader."""
        if self.test_ds is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        
        return DataLoader(
            self.test_ds,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )
    
    def _build_gpu_aug_global(self) -> v2.Compose:
        """Build GPU augmentation pipeline for global views."""
        return v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _build_gpu_aug_local(self) -> v2.Compose:
        """Build GPU augmentation pipeline for local views."""
        return v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def configure_optimizers(self):
        """Lightning hook for optimizer configuration."""
        # Use single optimizer with parameter groups to support automatic optimization
        # and gradient clipping while having different learning rates
        optimizer = torch.optim.AdamW([
            {
                'params': self.encoder.parameters(),
                'lr': self.config.lr,
                'weight_decay': self.config.weight_decay,
                'betas': (0.9, 0.95),
            },
            {
                'params': self.probe.parameters(),
                'lr': 3e-3,
                'weight_decay': 0.0,
                'betas': (0.9, 0.95),
            }
        ])
        
        # Match run_JEPA.py scheduler: 1-epoch linear warmup + cosine decay
        # Calculate steps per epoch (accounting for gradient accumulation)
        steps_per_epoch = len(self.train_ds) // self.per_device_batch_size // self.config.grad_accum
        warmup_steps = steps_per_epoch  # 1 epoch warmup
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps, 
            eta_min=1e-6
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
                'interval': 'step',  # Step every batch, not epoch
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
        # Note: In Lightning, gathering outputs across batches is more complex
        # For simplicity, we'll skip the detailed metrics for now
        # You can implement this using self.trainer.callback_metrics if needed
        pass
    
    def on_save_checkpoint(self, checkpoint):
        """Lightning hook for saving additional state."""
        checkpoint['best_acc'] = self.best_acc
        checkpoint['method_name'] = self.get_method_name()


# ========== Concrete Trainer Implementations ==========

class JEPATrainer(BaseTrainer):
    """Trainer for LeJEPA / SIGReg based methods."""
    
    def __init__(
        self,
        encoder: nn.Module,
        config: TrainerConfig,
        lamb: float = 0.05,
        hydra_cfg: Optional[Any] = None,
    ):
        super().__init__(encoder, config, hydra_cfg)
        self.sigreg = SIGReg()
        self.lamb = lamb
    
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
        
        # Gather across GPUs if using DDP (Lightning auto-detects)
        if self.config.distributed and self.config.reg == "hybrid":
            gathered_proj = self.all_gather(all_proj, sync_grads=True)
            gathered_emb = self.all_gather(all_emb, sync_grads=True)  # ADD THIS
            all_proj = gathered_proj.flatten(0, 1)
            all_emb = gathered_emb.flatten(0, 1)  # ADD THIS
            labels = self.all_gather(labels, sync_grads=False).flatten(0, 1)

        # Now all_emb, all_proj, and labels are all consistent
        global_proj = all_proj[:, :self.config.V_global, :] # Vg
        

        # Get global step (Lightning provides this)
        global_step = self.global_step if hasattr(self, 'global_step') else 0
        
        ssl_loss, pred_loss, sigreg_loss = LeJEPA(
            global_proj, all_proj, self.sigreg, self.lamb, 
            losstype=self.config.reg, labels=labels, global_step=global_step
        )
        if self.config.reg == "hybrid":
            local_proj = all_proj[:, self.config.V_global:, :]
            ssl_loss += simclr_loss(global_proj, local_proj)

        # Compute probe loss
        V = self.config.V_global + self.config.V_local+self.config.V_mixed
        y_rep = labels.repeat_interleave(V)
        yhat = self.probe(all_emb.flatten(0, 1))
        probe_loss = F.cross_entropy(yhat, y_rep)

        total_loss = ssl_loss + probe_loss
        return {
            "total_loss": total_loss,
            "sigreg_loss": sigreg_loss,
            "prediction_loss": pred_loss,
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
            gathered_emb = self.all_gather(all_emb, sync_grads=True)  # UNCOMMENT
            
            all_proj = gathered_proj.flatten(0, 1)
            all_emb = gathered_emb.flatten(0, 1)  # UNCOMMENT
            labels = gathered_labels.flatten(0, 1)
            
        global_proj = all_proj[:, :self.config.V_global, :]
        local_proj = all_proj[:, self.config.V_global:, :]
        
        # SimCLR loss
        cl_loss = simclr_loss(global_proj, local_proj, labels, temperature=self.temperature)
        # Compute probe loss
        V = self.config.V_global + self.config.V_local+self.config.V_mixed
        y_rep = labels.repeat_interleave(V)
        yhat = self.probe(all_emb.flatten(0, 1))
        probe_loss = F.cross_entropy(yhat, y_rep)
        total_loss = cl_loss + probe_loss
        

        return {
            "total_loss": total_loss,
            "cl_loss": cl_loss,
            "probe_loss": probe_loss,
        }
