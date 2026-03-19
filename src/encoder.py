import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP

# ── Paper model presets (Table 2) ───────────────────────────────────────────
MODEL_VIT_L = "vit_large_patch16_224"        # ~304M params
MODEL_CONVNEXTV2_H = "convnextv2_huge"      # ~660M params


def _is_vit(name: str) -> bool:
    return name.startswith("vit")


class Encoder(nn.Module):
    """
    Backbone + projection head for LeJEPA.

    Supports both Vision-Transformer and ConvNeXt families from timm.
    ViT-specific options (dynamic_img_size, reg_tokens) are applied only
    when the backbone is a ViT variant.
    """
    def __init__(self, model_name=MODEL_VIT_L, proj_dim=512):
        super().__init__()
        
        cfg = {
            "pretrained": False,
            "num_classes": 0,
            "drop_path_rate": 0.1,
        }

        if _is_vit(model_name):
            cfg["dynamic_img_size"] = True
            cfg["reg_tokens"] = 0  # paper: no register tokens

        self.backbone = timm.create_model(model_name, **cfg)
        self.backbone.set_grad_checkpointing(True)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.backbone = torch.compile(
            self.backbone,
            mode="reduce-overhead",
        )

        self.feat_dim = self.backbone.num_features
        self.proj = MLP(
            in_channels=self.feat_dim, 
            hidden_channels=[2048, 2048, proj_dim], 
            norm_layer=nn.BatchNorm1d
        )
        self.proj = torch.compile(self.proj, mode="default")
        self.output_bn = nn.BatchNorm1d(proj_dim, affine=False)
        self.output_bn = nn.SyncBatchNorm.convert_sync_batchnorm(self.output_bn)
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
