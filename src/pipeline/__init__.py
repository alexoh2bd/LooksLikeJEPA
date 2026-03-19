"""B3-LeJEPA batch mining pipeline for self-supervised JEPA pretraining.

Exposes the ``LeJEPABatchSampler`` at the package top level for convenience.

    from src.pipeline import LeJEPABatchSampler
"""

from pipeline.batch_sampler import LeJEPABatchSampler

__all__ = ["LeJEPABatchSampler"]
