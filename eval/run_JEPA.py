import torch
import torch.profiler
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
from datasets import load_dataset
from torch.amp import  autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from save import save_checkpoint
from stats import  RepresentationMetrics
from loss import SIGReg, LeJEPA
from ds import HFDataset, CrossInstanceDataset, collate_views


# Setup logging
logging.basicConfig(level=logging.INFO)


class Encoder(nn.Module):
    """
    Encoder for ViT.
    """
    def __init__(self, model_name="vit_base_patch16_224.dino", proj_dim=512):
        super().__init__()
        
        cfg = {
            "pretrained": False,
            "num_classes": 0,
            "drop_path_rate": 0.1,
            "dynamic_img_size": True if model_name.startswith("vit") else False
        }


        self.backbone = timm.create_model(
            model_name,
            **cfg,
        )

        # Memory optimization: Enable gradient checkpointing to trade compute for memory
        # Saves ~20-40 GB by not storing all intermediate activations
        self.backbone.set_grad_checkpointing(True)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Convert backbone to bfloat16 BEFORE compiling to avoid dtype mismatch
        # (conv2d error: "Input type BFloat16 and bias type float should be the same")
        # self.backbone = self.backbone.to(torch.bfloat16)
        
        self.backbone = torch.compile(
            self.backbone,
            mode="default",
        )

        self.feat_dim = self.backbone.num_features
        self.proj = MLP(
            in_channels=self.feat_dim, 
            hidden_channels=[2048, 2048, proj_dim], 
            norm_layer=nn.BatchNorm1d
        )
        self.proj = torch.compile(self.proj, mode="default")
        self.output_bn = nn.BatchNorm1d(proj_dim, affine=False)
        self.output_bn.float()

    def forward(self, x_list, unnorm=False):
        """
        x_list: List of tensors. Each tensor is (B, C, H, W).
                Can contain any mix of resolutions.
        """
        # #region agent log
        # #endregion
        idx_global = [i for i, t in enumerate(x_list) if t.shape[-1] >= 128]
        idx_local = [i for i, t in enumerate(x_list) if t.shape[-1] < 128]
        
        results = {}

        # Process Global Views
        if len(idx_global) > 0:
            # (N, Vg, C, H, W)
            g_imgs = torch.stack([x_list[i] for i in idx_global], dim=1)
            N, Vg, C = g_imgs.shape[:3]
            
            # #region agent log
            # #endregion
            
            # Forward Pass
            # backbone(flattened) -> (N*Vg, feature_dim)
            emb_g = self.backbone(g_imgs.flatten(0, 1)) 
            proj_g = self.proj(emb_g) # (N*Vg, proj_dim)
            proj_g = self.output_bn(proj_g)
            
            # Reshape back to (N, Vg, D)
            proj_g = proj_g.reshape(N, Vg, -1)
            emb_g = emb_g.reshape(N, Vg, -1)
            
            # Store results back in the dictionary
            for k, original_idx in enumerate(idx_global):
                results[original_idx] = (emb_g[:, k], proj_g[:, k])

        # Process Local Views
        if len(idx_local) > 0:
            l_imgs = torch.stack([x_list[i] for i in idx_local], dim=1)
            N, Vl, C = l_imgs.shape[:3]
            
            # backbone(flattened) -> (N*Vl, feature_dim)
            emb_l = self.backbone(l_imgs.flatten(0, 1))
            proj_l = self.proj(emb_l) # (N*Vl, proj_dim)
            proj_l = self.output_bn(proj_l)
            
            # Reshape back to (N, Vl, D)
            proj_l = proj_l.reshape(N, Vl, -1)
            emb_l = emb_l.reshape(N, Vl, -1)
            
            for k, original_idx in enumerate(idx_local):
                results[original_idx] = (emb_l[:, k], proj_l[:, k])

        # Reassemble in original order
        final_emb = []
        final_proj = []
        
        for i in range(len(x_list)):
            e, p = results[i]
            final_emb.append(e)
            final_proj.append(p)
        
        # Stack to return (N, V, D)
        return torch.stack(final_emb, dim=1), torch.stack(final_proj, dim=1)

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # Crucial for deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    logging.info(cfg)

    # Hyperparameters
    num_workers = cfg.get("num_workers", 4)  # FIXED: Use .get() with defaults
    proj_dim = cfg.get("proj_dim", 128)  # FIXED: Increased to match feature_dim
    lr = cfg.get("lr", 1e-3)  # FIXED: Lower initial LR for ViT
    bs = cfg.get("bs", 256)
    epochs = cfg.get("epochs", 100)
    lamb = cfg.get("lamb", 0.05)  # FIXED: Much lower lambda
    V_global = cfg.get("V_global", 2)
    V_local = cfg.get("V_local", 4)
    V_mixed = cfg.get("V_mixed", 1)
    grad_accum = cfg.get("grad_accum", 1)  # Gradient accumulation steps
    view_selection = "random" if V_mixed == 0 else "mixed"

    device = cfg.get("device", "cuda")
    prefetch_factor = cfg.get("prefetch_factor", 2)
    model_name = cfg.get("model_name", "vit_base_patch16_224.dino")
    global_img_size = cfg.get("global_img_size", 224)
    local_img_size = cfg.get("local_img_size", 96)
    dataset = cfg.get("dataset", "inet100")
    reg =cfg.get("reg", "LeJEPA")
    save_prefix = f"{reg}_{dataset}_{model_name}/LV{V_local}|MV{V_mixed}|BS{bs*grad_accum}_e{epochs}"
    wandb.init(project="VIT_JEPA_Views", name = save_prefix,config=dict(cfg))

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    
    # Datasets
    if view_selection == "mixed" or view_selection == "hybrid":
        train_ds = CrossInstanceDataset("train", V_global=V_global, V_local=V_local, V_mixed = V_mixed, local_img_size=local_img_size, global_img_size=global_img_size, dataset=dataset)
    elif view_selection == "random":
        train_ds = HFDataset("train", V_global=V_global, V_local=V_local, local_img_size=local_img_size, global_img_size=global_img_size, dataset=dataset)
    test_split = "test" if dataset == "cifar10" else "val"
    test_ds = HFDataset(test_split, V_global=1, V_local=0, local_img_size=local_img_size, global_img_size=global_img_size, dataset=dataset)
    
    # Configure View Counts
    # if view_selection == "mixed":
    #     train_ds.n_global = V_global
    #     train_ds.n_local = 12
    # else:
    train_ds.n_global = V_global
    train_ds.n_local = V_local
    train_ds.global_img_size = global_img_size
    train_ds.local_img_size = local_img_size

         
    # Data Loaders
    train = DataLoader(
        train_ds, 
        batch_size=bs, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=num_workers > 0,  # FIXED: Only if workers exist
        prefetch_factor=prefetch_factor if num_workers > 0 else None,  # FIXED: Only with workers
        collate_fn=collate_views,  # Pre-stack views by resolution for efficient GPU transfer
    )
    test = DataLoader(
        test_ds, 
        batch_size=bs, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Model Setup
    net = Encoder(model_name=model_name, proj_dim=proj_dim).to(device)
    
    num_classes=100 if dataset =="inet100" else 10
    probe = nn.Sequential(
        nn.LayerNorm(net.feat_dim), 
        nn.Linear(net.feat_dim, num_classes)
    ).to(device)
    sigreg = SIGReg().to(device)
    # sigreg = EppsPulley().to(device)
   
    opt_encoder = torch.optim.AdamW(
        net.parameters(), 
        lr=lr, 
        weight_decay=5e-2,
        betas=(0.9, 0.95)  # FIXED: ViT-friendly betas
    )
    
    opt_probe = torch.optim.AdamW(
        probe.parameters(), 
        lr=3e-3, 
        weight_decay=0.0
    )

    # Schedulers - account for gradient accumulation
    # With grad_accum, we have len(train) / grad_accum optimizer steps per epoch
    steps_per_epoch = len(train) // grad_accum
    warmup_steps = steps_per_epoch  # 1 epoch warmup in optimizer steps
    total_steps = steps_per_epoch * epochs
    s1 = LinearLR(opt_encoder, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt_encoder, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(opt_encoder, schedulers=[s1, s2], milestones=[warmup_steps])
    
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
    bestacc = 0

 

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

                all_views = global_views + local_views 
                all_emb, all_proj = net(all_views)    
                global_proj= all_proj[:, :len(global_views), :]
                reg_loss, pred_loss, sigreg_loss = LeJEPA(global_proj, all_proj, sigreg, lamb, losstype=reg, labels=y, global_step=global_step)
                y_rep = y.repeat_interleave(len(vs)) # (N*V,)
                yhat = probe(all_emb.flatten(0, 1).detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                
                loss = reg_loss + probe_loss
                
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
                    "train/probe_loss": probe_loss.item(),
                    "train/lejepa_loss": reg_loss.item(),
                    "train/sigreg_loss": sigreg_loss.item(),
                    "train/prediction_loss": pred_loss.item(),
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
                    # Since V=1 in test, this effectively squeezes the view dim
                    emb_flat = emb.flatten(0, 1) # (N*V, D)
                    # logging.info(f"flat emb shape: {emb_flat.shape}")
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
        
    save_checkpoint(cfg, net, probe, opt_encoder, opt_probe, epoch, global_step, acc, reg, dataset, V=V_local, save_prefix=save_prefix)

    wandb.finish()

if __name__ == "__main__":
    main()