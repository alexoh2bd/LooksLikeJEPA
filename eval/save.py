import os
import torch
import logging

def save_checkpoint(cfg, net, probe, opt_enc, opt_probe, epoch, step, acc, reg="LeJEPA", dataset="cifar10", V=4, save_prefix="base"):
    # 1. Create directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), f"data/checkpoints/{save_prefix}/V{V}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Prepare the State Dictionary
    # We save everything needed to resume training exactly where we left off
    state = {
        'epoch': epoch,
        'step': step,
        'config': dict(cfg), # Save hyperparameters for reference
        'best_acc': acc,
        
        # Models
        'encoder': net.state_dict(),       # Contains backbone + projector
        'backbone_only': net.backbone.state_dict(), # Pure ViT (Useful for downstream!)
        'probe': probe.state_dict(),
        
        # Optimizers (Critical for resuming training)
        'opt_encoder': opt_enc.state_dict(),
        'opt_probe': opt_probe.state_dict(),
        
    }

    # 3. Save "Last" (Overwrites every epoch to save space)
    last_path = os.path.join(save_dir, f"checkpoint_last{reg}_e{epoch}_{dataset}_{V}LV.pth")
    torch.save(state, last_path)
    