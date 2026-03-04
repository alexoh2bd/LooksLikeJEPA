"""
Lightning-based Training Script for SSL methods (LeJEPA, SimCLR, etc.)

Usage (LeJEPA – Table 2 replication):
    python src/run_training_loop.py \\
        +reg=LeJEPA \\
        +model_name=vit_large_patch16_224 \\
        +dataset=imagenet-1k \\
        +epochs=100 +bs=256 \\
        +lr=5e-4 +weight_decay=1e-2 \\
        +V_global=2 +V_local=6 +V_mixed=0 \\
        +lamb=0.05 +use_swa=False
"""
import os
import torch
import logging
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning)
from trainer import TrainerConfig, SimCLRTrainer, JEPATrainer, LpJEPATrainer
from encoder import Encoder


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Set seed for reproducibility
    seed = cfg.get("seed", 0)
    L.seed_everything(seed, workers=True)
    # In run_training_loop.py, after L.seed_everything():
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # Disable for reproducibility
    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    
    logging.info(cfg)
    
    # Build config from hydra
    config = TrainerConfig.from_hydra(cfg)
    
    # Create encoder
    encoder = Encoder(model_name=config.model_name, proj_dim=config.proj_dim)
    
    # Create model based on method type
    reg = cfg.get("reg", "LeJEPA")
    if reg == "SimCLR" or reg=="SupCon":
        model = SimCLRTrainer(
            encoder=encoder,
            config=config,
            temperature=cfg.get("temperature", 0.5),
            hydra_cfg=cfg,
        )
    elif reg in ("LeJEPA", "hybrid", "weighted_hybrid"):
        model = JEPATrainer(
            encoder=encoder,
            config=config,
            lamb=cfg.get("lamb", 0.05),
            w=cfg.get("w", 0.5),
            hydra_cfg=cfg,
        )
    elif reg in ("LpJEPA", "lp_hybrid"):
        model = LpJEPATrainer(
            encoder=encoder,
            config=config,
            invariance_weight=cfg.get("invariance_weight", 25.0),
            rdm_reg_weight=cfg.get("rdm_reg_weight", 25.0),
            lp_norm_parameter=cfg.get("lp_norm_parameter", 1.0),
            mean_shift_value=cfg.get("mean_shift_value", 0.0),
            target_distribution=cfg.get("target_distribution", "rectified_lp_distribution"),
            num_projections=cfg.get("num_projections", 256),
            hydra_cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown method: {reg}")
    save_prefix = f"{model.get_method_name()}_{config.dataset}/LV{config.V_local}_MV{config.V_mixed}_BS{config.bs * config.grad_accum}_e{config.epochs}{'_ddp1' if config.distributed else ''}"
    logging.info(f"save_prefix: {save_prefix}")
    ckpt_dir=f"data/checkpoints/{save_prefix}"
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{val/acc:.3f}",
        monitor="val/acc",
        mode="max",
        save_top_k=2,
        save_last=True,
        every_n_epochs=None,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    wandb_logger = WandbLogger(
        project="VIT_JEPA_Views",
        name=save_prefix,
        config=dict(cfg),
    )
    
    # Create Lightning Trainer
    # Use "auto" for single GPU (avoids DDP overhead), "ddp" for multi-GPU
    world_size = cfg.get("world_size", 1)
    use_ddp = cfg.get("distributed", False) and world_size > 1
    num_nodes=cfg.get("num_nodes",1)
    ndevices = world_size //num_nodes
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=ndevices if use_ddp else 1,
        strategy="ddp" if use_ddp else "auto",
        precision="bf16-mixed",
        accumulate_grad_batches=config.grad_accum,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=config.log_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        deterministic=False,  # Disabled: ViT's bicubic upsampling is non-deterministic
        enable_progress_bar=True,
        num_nodes=num_nodes,
        use_distributed_sampler=use_ddp,
        sync_batchnorm=use_ddp,
    )
    last_ckpt=f"data/checkpoints/{save_prefix}/last.ckpt"

    torch.serialization.add_safe_globals([TrainerConfig])
    # Train the model
    trainer.fit(model, ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)

    if checkpoint_callback.best_model_path:
        logging.info(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")
    if checkpoint_callback.last_model_path:
        logging.info(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")

if __name__ == "__main__":
    main()
