def log_train(epoch, loss, lr):
    log_dict = {
        "train/probe_loss": probe_loss.item(),
        "train/lejepa_loss": lejepa_loss.item(),
        "train/sigreg_loss": sigreg_loss.item(),
        "train/prediction_loss": pred_loss.item(),
        "train/lr": opt_encoder.param_groups[0]["lr"],
        "train/epoch": epoch,
        "train/global_step": global_step,
    }
    
    # Expensive metrics - Compute less frequently to avoid GPU sync spikes
    if global_step % 500 == 0:
        with torch.no_grad():
            proj_std = all_proj.std(dim=(0, 1)).mean().item()
            proj_rank = torch.linalg.matrix_rank(all_proj.flatten(0, 1).float()).item()
            norm = all_proj.norm().item()
        
        log_dict.update({
            "train/proj_std": proj_std,
            "train/proj_rank": proj_rank, 
            "train/projected_norm": norm,
        })
    
    wandb.log(log_dict, step=global_step)