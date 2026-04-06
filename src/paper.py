"""Paper LeJEPA baseline: Lightning wrapper around stable_pretraining.methods.lejepa.LeJEPA.

Uses the library loss unchanged. DDP: runtime patches EppsPulley / SlicedEppsPulley forward
so dist checks run at forward time (init-time cache is often wrong under Lightning).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from ds import HFDataset, collate_paper, collate_paper_val
from stable_pretraining.backbone import MLP
from stable_pretraining.methods.lejepa import LeJEPA

logger = logging.getLogger(__name__)


def _patch_lejepa_epps_for_ddp() -> None:
    """Replace forward methods so DDP is detected at forward time, not __init__."""
    import torch.distributed as dist
    from stable_pretraining.methods import lejepa as lejepa_mod
    from torch.distributed.nn import all_reduce as dist_nn_all_reduce

    def epps_forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)
        is_ddp = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_ddp else 1
        if is_ddp:
            dist_nn_all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            dist_nn_all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * world_size

    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            step = self.global_step.clone()
            is_ddp = dist.is_available() and dist.is_initialized()
            if is_ddp:
                torch.distributed.broadcast(step, src=0)
            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)
        proj = x @ A
        return self.ep(proj).mean()

    lejepa_mod.EppsPulley.forward = epps_forward
    lejepa_mod.SlicedEppsPulley.forward = sliced_forward
    logger.info("Applied runtime DDP patches to EppsPulley / SlicedEppsPulley.forward")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _num_classes_for_dataset(dataset: str) -> int:
    return {"imagenet-1k": 1000, "inet100": 100, "cifar10": 10}.get(dataset, 1000)


class PaperTrainer(L.LightningModule):
    """SSL pretraining with official stable-pretraining LeJEPA (inv + lamb * SIGReg)."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        _patch_lejepa_epps_for_ddp()

        encoder_name = _cfg_get(cfg, "model_name", "vit_large_patch16_224")
        lamb = float(_cfg_get(cfg, "lamb", 0.02))
        n_slices = int(_cfg_get(cfg, "sigreg_n_slices", 1024))
        t_max = float(_cfg_get(cfg, "sigreg_t_max", 3.0))
        n_points = int(_cfg_get(cfg, "sigreg_n_points", 17))
        drop_path_rate = float(_cfg_get(cfg, "drop_path_rate", 0.1))
        proj_dim = int(_cfg_get(cfg, "proj_dim", 512))

        projector: Optional[nn.Module] = None
        if proj_dim != 512:
            bb = timm.create_model(
                encoder_name,
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate,
                **({"dynamic_img_size": True} if "vit" in encoder_name else {}),
            )
            embed_dim = bb.num_features
            del bb
            projector = nn.Sequential(
                nn.Linear(embed_dim, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, proj_dim],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )

        self.model = LeJEPA(
            encoder_name=encoder_name,
            projector=projector,
            n_slices=n_slices,
            t_max=t_max,
            n_points=n_points,
            lamb=lamb,
            pretrained=False,
            drop_path_rate=drop_path_rate,
        )

        ds_name = _cfg_get(cfg, "dataset", "imagenet-1k")
        feat_dim = self.model.embed_dim
        n_cls = _num_classes_for_dataset(ds_name)
        self.probe = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, n_cls),
        )

        self.train_ds = None
        self.val_ds = None
        self._train_sampler: Optional[DistributedSampler] = None

    def on_fit_start(self) -> None:
        """Align WandB charts on global_step (optional; no-op if not using WandB)."""
        if self.logger is None:
            return
        exp = getattr(self.logger, "experiment", None)
        if exp is None or not hasattr(exp, "define_metric"):
            return
        try:
            exp.define_metric("trainer/global_step")
            exp.define_metric("train/*", step_metric="trainer/global_step")
            exp.define_metric("lr/*", step_metric="trainer/global_step")
        except Exception as e:
            logger.debug("wandb define_metric skipped: %s", e)

    def get_method_name(self) -> str:
        return "paper"

    @property
    def distributed(self) -> bool:
        return bool(_cfg_get(self.cfg, "distributed", False))

    @property
    def world_size(self) -> int:
        return int(_cfg_get(self.cfg, "world_size", 1))

    @property
    def per_device_batch_size(self) -> int:
        bs = int(_cfg_get(self.cfg, "bs", 256))
        ws = self.world_size if self.distributed else 1
        pdb = bs // ws
        if pdb < 1:
            raise ValueError(f"Batch size {bs} too small for world_size={ws}")
        return pdb

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return
        cfg = self.cfg
        ds_name = _cfg_get(cfg, "dataset", "imagenet-1k")
        V_global = int(_cfg_get(cfg, "V_global", 2))
        V_local = int(_cfg_get(cfg, "V_local", 6))
        gsz = int(_cfg_get(cfg, "global_img_size", 224))
        lsz = int(_cfg_get(cfg, "local_img_size", 98))
        seed = int(_cfg_get(cfg, "seed", 0))
        test_split = "test" if ds_name == "cifar10" else "val"

        self.train_ds = HFDataset(
            "train",
            V_global=V_global,
            V_local=V_local,
            global_img_size=gsz,
            local_img_size=lsz,
            dataset=ds_name,
            seed=seed,
            paper_augmentations=True,
        )
        self.val_ds = HFDataset(
            test_split,
            V_global=1,
            V_local=0,
            global_img_size=gsz,
            local_img_size=lsz,
            dataset=ds_name,
            seed=seed,
            paper_augmentations=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("setup(fit) must run before train_dataloader")
        cfg = self.cfg
        nw = int(_cfg_get(cfg, "num_workers", 4))
        pf = int(_cfg_get(cfg, "prefetch_factor", 2))
        seed = int(_cfg_get(cfg, "seed", 0))
        V_global = int(_cfg_get(cfg, "V_global", 2))
        V_local = int(_cfg_get(cfg, "V_local", 6))

        collate_fn = lambda b: collate_paper(b, V_global, V_local)

        if self.distributed:
            sampler = DistributedSampler(
                self.train_ds,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
            self._train_sampler = sampler
            return DataLoader(
                self.train_ds,
                batch_size=self.per_device_batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=nw,
                pin_memory=True,
                persistent_workers=nw > 0,
                prefetch_factor=pf if nw > 0 else None,
                collate_fn=collate_fn,
            )

        self._train_sampler = None
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            self.train_ds,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=nw,
            pin_memory=True,
            generator=g,
            persistent_workers=nw > 0,
            prefetch_factor=pf if nw > 0 else None,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("setup(fit) must run before val_dataloader")
        cfg = self.cfg
        nw = int(_cfg_get(cfg, "num_workers", 4))
        pf = int(_cfg_get(cfg, "prefetch_factor", 2))
        sampler = None
        if self.distributed:
            sampler = DistributedSampler(self.val_ds, shuffle=False)
        return DataLoader(
            self.val_ds,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=pf if nw > 0 else None,
            collate_fn=collate_paper_val,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels = batch["label"]
        V_global = int(_cfg_get(self.cfg, "V_global", 2))

        out = self.model(
            global_views=batch["global_views"],
            local_views=batch["local_views"],
        )
        # Online linear probe on detached global backbone features [N*Vg, D]
        y_rep = labels.repeat(V_global)
        probe_loss = F.cross_entropy(self.probe(out.embedding), y_rep)
        ssl_loss = out.loss
        total_loss = ssl_loss + probe_loss

        sync = True
        self.log(
            "train/ssl_loss",
            ssl_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=sync,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/probe_loss",
            probe_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=sync,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=sync,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/inv_loss",
            out.inv_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=sync,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/sigreg_loss",
            out.sigreg_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=sync,
            prog_bar=False,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        out = self.model(images=batch["images"])
        logits = self.probe(out.embedding)
        acc = (logits.argmax(1) == batch["label"]).float().mean()
        self.log(
            "val/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        gn = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log(
            "train/grad_norm",
            gn,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )

    def configure_optimizers(self):
        cfg = self.cfg
        lr = float(_cfg_get(cfg, "lr", 5e-4))
        probe_lr = float(_cfg_get(cfg, "probe_lr", 3e-3))
        model_name = str(_cfg_get(cfg, "model_name", "vit_large_patch16_224")).lower()
        wd_override = _cfg_get(cfg, "weight_decay", None)
        if wd_override is not None:
            encoder_wd = float(wd_override)
        else:
            encoder_wd = 5e-2 if model_name.startswith("vit") else 5e-4
        epochs = int(_cfg_get(cfg, "epochs", 100))
        warmup_epochs = int(_cfg_get(cfg, "warmup_epochs", 10))
        final_lr_ratio = float(_cfg_get(cfg, "final_lr_ratio", 1e-3))

        opt = AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "lr": lr,
                    "weight_decay": encoder_wd,
                    "betas": (0.9, 0.999),
                },
                {
                    "params": self.probe.parameters(),
                    "lr": probe_lr,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.999),
                },
            ],
        )

        warmup = LinearLR(
            opt,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        cosine_epochs = max(1, epochs - warmup_epochs)
        cosine = CosineAnnealingLR(
            opt,
            T_max=cosine_epochs,
            eta_min=lr * final_lr_ratio,
        )
        sched = SequentialLR(
            opt,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.current_epoch)
