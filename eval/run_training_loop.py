"""
Lightning-based Training Script for SSL methods (LeJEPA, SimCLR, etc.)

Usage:
    python eval/run_training_loop.py +bs=256 +epochs=100 +dataset=inet100 ...
"""
import torch
import logging
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from trainer import TrainerConfig, SimCLRTrainer, JEPATrainer
from run_JEPA import Encoder


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Set seed for reproducibility
    seed = cfg.get("seed", 0)
    L.seed_everything(seed, workers=True)
    
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
    elif reg == "LeJEPA" or reg == "hybrid":
        model = JEPATrainer(
            encoder=encoder,
            config=config,
            lamb=cfg.get("lamb", 0.05),
            hydra_cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown method: {reg}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{model.get_method_name()}_{config.dataset}",
        filename="{epoch}-{val/acc:.3f}",
        monitor="val/acc",
        mode="max",
        save_top_k=0,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    save_prefix = f"{model.get_method_name()}_{config.dataset}_{config.model_name}/LV{config.V_local}|MV{config.V_mixed}|BS{config.bs * config.grad_accum}_e{config.epochs}"
    wandb_logger = WandbLogger(
        project="VIT_JEPA_Views",
        name=save_prefix,
        config=dict(cfg),
    )
    
    # Create Lightning Trainer
    # Use "auto" for single GPU (avoids DDP overhead), "ddp" for multi-GPU
    world_size = cfg.get("world_size", 1)
    use_ddp = cfg.get("distributed", False) and world_size > 1
    
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=world_size if use_ddp else 1,
        strategy="ddp" if use_ddp else "auto",
        precision="bf16-mixed",
        accumulate_grad_batches=config.grad_accum,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=config.log_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        deterministic=False,  # Disabled: ViT's bicubic upsampling is non-deterministic
        enable_progress_bar=True,
        use_distributed_sampler=use_ddp,
        sync_batchnorm=use_ddp,
    )
    
    # Train the model
    trainer.fit(model)


if __name__ == "__main__":
    main()
