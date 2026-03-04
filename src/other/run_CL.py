''' 
Implementation based on https://github.com/sthalles/SimCLR by Thalles Silva
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm
import wandb
import hydra
import tqdm
import logging
from omegaconf import DictConfig
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torchvision.ops import MLP
from loss import  simclr_loss
from stats import RepresentationMetrics
from save import save_checkpoint
from ds import HFDataset, CrossInstanceDataset
from run_JEPA import Encoder


logging.basicConfig(level=logging.INFO)



@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # GPU settings
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.manual_seed(0)
    logging.info(cfg)

    # Hyperparameters
    device = cfg.get("device", "cuda")
    bs = cfg.get("bs", 512)
    epochs = cfg.get("epochs", 300)
    lr = cfg.get("lr", 5e-4)
    num_workers = cfg.get("num_workers", 4)
    model_name = cfg.get("model_name", "vit_base_patch16_224.dino")
    proj_dim = cfg.get("proj_dim", 512)
    dataset = cfg.get("dataset", "cifar10")
    grad_accum = cfg.get("grad_accum", 4)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    global_img_size = cfg.get("global_img_size", 224)
    local_img_size = cfg.get("local_img_size", 96)
    temperature = cfg.get("temperature", 0.5)
    V_global = cfg.get("V_global", 2)
    V_local = cfg.get("V_local", 4)
    V_mixed =cfg.get("V_mixed", 2)
    view_selection = "random" if V_mixed == 0 else "mixed"
    reg="SimCLR"


    save_prefix = f"SimCLR_{dataset}_{model_name}/LV{V_local}_e{epochs}"
    wandb.init(project="VIT_JEPA_Views",name=save_prefix, config=dict(cfg))
    class100 =set[str](["inet100"])
    num_classes = 100 if dataset in class100 else 10
    
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # Data
    if view_selection == "mixed":
        train_ds = CrossInstanceDataset("train", V_global=V_global, V_local=V_local, V_mixed = V_mixed, local_img_size=local_img_size, global_img_size=global_img_size, dataset=dataset)
    elif view_selection == "random":
        train_ds = HFDataset("train", V_global=V_global, V_local=V_local, local_img_size=local_img_size, global_img_size=global_img_size, dataset=dataset)
    test_split = "test" if dataset == "cifar10" else "val"

    test_ds = HFDataset(test_split, dataset=dataset)

    train = DataLoader(
        train_ds, 
        batch_size=bs, 
        shuffle=True, 
        drop_last=True,
        # collate_fn = contrastive_collate_fn,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test = DataLoader(
        test_ds, 
        batch_size=bs, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    # Model
    net = Encoder(model_name, proj_dim).to(device)
    probe = nn.Sequential(
        nn.LayerNorm(net.feat_dim), 
        nn.Linear(net.feat_dim, num_classes)
    ).to(device)
    
    opt_probe = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    opt_encoder = torch.optim.AdamW(
        net.parameters(), 
        lr=lr, 
        weight_decay=5e-2,
        betas=(0.9, 0.95)  
    )

    # Scheduler
    steps_per_epoch = len(train) // grad_accum
    warmup = steps_per_epoch  # 1 epoch warmup in optimizer steps
    total_steps = steps_per_epoch * epochs
    s1 = LinearLR(opt_encoder, start_factor=0.01, total_iters=warmup)
    s2 = CosineAnnealingLR(opt_encoder, T_max=total_steps - warmup, eta_min=1e-6)
    scheduler = SequentialLR(opt_encoder, [s1, s2], milestones=[warmup])
    
    # GPU augmentations
    gpu_aug_global = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
        v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2), 
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    gpu_aug_local = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)
    repmetrics = RepresentationMetrics()
    global_step = 0
    
    for epoch in range(epochs):
        net.train()
        probe.train()
        logging.info(f"Epoch {epoch}")
        
        # Zero gradients at start of epoch for gradient accumulation
        opt_encoder.zero_grad()
        opt_probe.zero_grad()
        
        for batch_idx, (vs, y) in enumerate(tqdm.tqdm(train, total=len(train))):
            
            with autocast(device, dtype=torch.bfloat16):
                # Move to device
                vs = [v.to(device, non_blocking=True) for v in vs]
                y = y.to(device, non_blocking=True)
                
                global_views = [vs[i] for i in range(len(vs)) if vs[i].shape[-1] == global_img_size]
                local_views = [vs[i] for i in range(len(vs)) if vs[i].shape[-1] == local_img_size]
                
                global_views = [gpu_aug_global(g) for g in global_views]
                local_views = [gpu_aug_local(l) for l in local_views]

                # Forward

                # net(global views) -> (N, Vg, D), (N, Vg, D)
                global_emb, global_proj = net(global_views)
                # net(local views) -> (N, Vl, D), (N, Vl, D)
                local_emb, local_proj = net(local_views)
                all_emb = torch.cat([global_emb, local_emb], dim=1)
                
                CL_loss = simclr_loss(global_proj, local_proj, y, temperature=temperature)
                y_rep = y.repeat_interleave(len(vs)) # (N*V,)
                yhat = probe(all_emb.flatten(0, 1).detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                
                loss = CL_loss + probe_loss
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum

            # Backward OUTSIDE autocast
            loss.backward()

            # Only step optimizers every grad_accum batches
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt_encoder.step()
                opt_probe.step()
                scheduler.step()
                
                opt_encoder.zero_grad()
                opt_probe.zero_grad()
                
            if global_step % 50 == 0:
                log_dict = {
                    "train/CL_loss": CL_loss.item(),
                    "train/probe_loss": probe_loss.item(),
                    "train/total_loss": (CL_loss + probe_loss).item(),  # Log unscaled loss
                    "train/lr": opt_encoder.param_groups[0]["lr"],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                wandb.log(log_dict, step=global_step)

            global_step += 1
        
        # Evaluation
        net.eval()
        probe.eval()
        correct = 0
        total_samples = 0
        all_test_emb = []
        all_test_labels = []

        with torch.inference_mode():
            for vs, y in test:
                vs = [v.to(device, non_blocking=True) for v in vs]
                y = y.to(device, non_blocking=True)
                
                with autocast(device, dtype=torch.bfloat16):
                    emb, _ = net(vs)
                    # Flatten (N, V, D) -> (N*V, D) before probe
                    emb_flat = emb.flatten(0, 1) # (N*V, D)
                    logits = probe(emb_flat) # (N*V, num_classes)
                    
                    correct += (logits.argmax(1) == y).sum().item()
                    total_samples += y.size(0)
                    
                    all_test_emb.append(emb_flat.detach()) # Keep on GPU for speed
                    all_test_labels.append(y)

        acc = correct / total_samples

        # Concatenate on GPU
        full_emb = torch.cat(all_test_emb, dim=0) # (Total_Test_Samples, D)
        full_labels = torch.cat(all_test_labels, dim=0) # (Total_Test_Samples,)


        
        # Compute metrics
        # align, uniformity = repmetrics.alignment_uniformity(full_emb)
        effective_rank = repmetrics.effective_rank(full_emb)
        fisher_ratio = repmetrics.fisher_ratio(full_emb, full_labels)
        cluster_metrics = repmetrics.cluster_quality_metrics(full_emb, full_labels)
        
        k_lid = min(20, full_emb.shape[0] // 10)
        mean_lid, lid_per_point = repmetrics.local_intrinsic_dimensionality(full_emb, k=k_lid)
        uniformity = repmetrics.uniformity(full_emb.to(device))
        wandb.log({
            # "test/alignment": align,
            "test/uniformity": uniformity, 
            "test/effective_rank": effective_rank, 
            "test/fisher_ratio": fisher_ratio, 
            "test/cluster_silhouette_score": cluster_metrics['silhouette'],
            "test/cluster_davies_bouldin_index": cluster_metrics['davies_bouldin'],
            "test/mean_lid": mean_lid,
            "test/acc": acc,
            "test/epoch": epoch
        }, step=global_step)
        if epoch+1 % 50 ==0:
            save_checkpoint(cfg, net, probe, opt_encoder, opt_probe, epoch, global_step, acc,  dataset, reg="SimCLR", V=V_local, save_prefix=save_prefix)
    wandb.finish()


if __name__ == "__main__":
    main()
