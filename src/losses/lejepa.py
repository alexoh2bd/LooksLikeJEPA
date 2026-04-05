"""LeJEPA encoder adapter for use inside JEPATrainer.

``LeJEPA`` is a thin ``nn.Module`` wrapper around
``stable_pretraining.methods.lejepa.LeJEPA``.  It borrows the backbone and
projector that the original initialises (architecture is byte-for-byte
identical), but:

* Inherits ``nn.Module`` (not ``pl.LightningModule``) — safe to nest inside
  ``JEPATrainer`` without stacking two Lightning modules.
* Replaces ``sigreg`` with a runtime-DDP-safe ``SlicedEppsPulley`` that
  checks ``dist.is_initialized()`` at forward-time rather than __init__.
* Exposes ``feat_dim`` alias required by ``BaseTrainer._build_probe``.
* Enables gradient checkpointing on the backbone.

``EppsPulley`` and ``SlicedEppsPulley`` mirror the originals in
``stable_pretraining.methods.lejepa`` with the DDP init-time caching fix.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.nn import all_reduce

from stable_pretraining.methods.lejepa import (  # type: ignore[import-untyped]
    LeJEPA as _StableLeJEPA,
    LeJEPAOutput,
)

__all__ = ["EppsPulley", "SlicedEppsPulley", "LeJEPAOutput", "LeJEPA"]


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Checks DDP status at forward-time so it works when instantiated before
    ``dist.init_process_group()`` is called by Lightning.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_ddp else 1

        if is_ddp:
            all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Checks DDP status at forward-time so the broadcast fires correctly under
    Lightning DDP (instantiated before dist.init_process_group).
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            step = self.global_step.clone()
            is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
            if is_ddp:
                torch.distributed.broadcast(step, src=0)
            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = x @ A
        return self.ep(proj).mean()


class LeJEPA(nn.Module):
    """nn.Module adapter wrapping ``stable_pretraining.methods.lejepa.LeJEPA``.

    The backbone and projector are initialised by the original class so the
    architecture is identical.  Three differences from the original:

    1. Base class is ``nn.Module`` — safe to nest inside ``JEPATrainer``.
    2. ``sigreg`` is replaced with a runtime-DDP-safe ``SlicedEppsPulley``.
    3. ``feat_dim`` alias and gradient checkpointing are added.
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        proj_dim: int = 512,
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        # Instantiate the original to reuse its backbone + projector setup
        _impl = _StableLeJEPA(
            encoder_name=encoder_name,
            n_slices=n_slices,
            t_max=t_max,
            n_points=n_points,
            lamb=lamb,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )
        self.backbone = _impl.backbone
        self.projector = _impl.projector if projector is None else projector
        self.embed_dim = _impl.embed_dim
        del _impl  # release the Lightning wrapper; backbone/projector now owned by self

        # Runtime-DDP-safe replacement for the stale-init sigreg
        self.sigreg = SlicedEppsPulley(num_slices=n_slices, t_max=t_max, n_points=n_points)
        self.lamb = lamb

        self.feat_dim = self.embed_dim  # alias for BaseTrainer._build_probe
        self.backbone.set_grad_checkpointing(True)

    # Reuse the original's static loss function without duplicating source
    _compute_loss = staticmethod(_StableLeJEPA._compute_loss)

    def forward(
        self,
        global_views: Optional[list] = None,
        local_views: Optional[list] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
        if self.training:
            assert global_views is not None and local_views is not None, (
                "global_views and local_views must be provided in training mode"
            )
            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))

            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            bs = global_views[0].shape[0]
            n_views = len(global_views) + len(local_views)
            all_projected = all_projected.view(n_views, bs, -1)

            loss, inv_loss, sigreg_loss = self._compute_loss(
                all_projected, len(global_views), self.sigreg, self.lamb
            )
            embedding = g_features.detach()
            return LeJEPAOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return LeJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sigreg_loss=zero,
            )
