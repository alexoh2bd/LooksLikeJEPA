"""
PCA Feature Visualization — Replicating LeJEPA Figure 14

For each input image:
1. Forward pass through the frozen ViT backbone (encoder only, no projector)
2. Extract last-layer patch tokens (exclude CLS token)
3. Run PCA on the (N_patches x D) matrix for that image independently
4. Map the first 3 principal components → R, G, B
5. Min-max normalize per channel to [0, 1]
6. Reshape to spatial grid and overlay/display alongside the original image

Usage:
    python pca_feature_viz.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --image_dir /path/to/images/ \
        --output_dir ./pca_viz_output \
        --arch vit_large_patch14_224 \
        --img_size 224 \
        --num_images 16

Notes:
    - This uses the ENCODER (backbone) features, not the projector.
    - PCA is computed per-image independently (not across the batch).
    - The technique follows DINO (Caron et al., 2021) and is used in
      LeJEPA Figure 14 for ViT-Large pretrained 100 epochs on ImageNet-1K.
"""

import argparse
import os
import glob
import sys

# Add src to path so Lightning checkpoint can unpickle (references 'trainer', etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import math

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from sklearn.decomposition import PCA

import timm


def parse_args():
    parser = argparse.ArgumentParser(description="PCA feature visualization (LeJEPA Fig 14 style)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint (.ckpt or .pth)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of images to visualize (uses glob for jpg/png)")
    parser.add_argument("--image_paths", type=str, nargs="+", default=None,
                        help="Explicit list of image paths")
    parser.add_argument("--output_dir", type=str, default="./pca_viz_output")
    parser.add_argument("--arch", type=str, default="vit_large_patch14_224",
                        help="timm model architecture name")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14,
                        help="Patch size (used to compute spatial grid dimensions)")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Max number of images to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Options for controlling the visualization ---
    parser.add_argument("--use_foreground_threshold", action="store_true",
                        help="Apply DINO-style foreground/background thresholding on PC1")
    parser.add_argument("--fg_threshold", type=float, default=0.0,
                        help="Threshold on first principal component to separate fg/bg. "
                             "Patches with PC1 > threshold are foreground.")
    parser.add_argument("--interpolation", type=str, default="nearest",
                        choices=["nearest", "bilinear"],
                        help="Interpolation for upscaling the PCA map to image resolution")
    
    # --- Checkpoint loading options ---
    parser.add_argument("--checkpoint_key", type=str, default=None,
                        help="Key in the checkpoint dict that holds the state_dict "
                             "(e.g., 'state_dict', 'model', 'encoder'). "
                             "If None, tries common keys automatically.")
    parser.add_argument("--strip_prefix", type=str, default=None,
                        help="Prefix to strip from state_dict keys "
                             "(e.g., 'encoder.', 'backbone.', 'module.')")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# 1. Model loading
# ──────────────────────────────────────────────────────────────

def load_encoder(arch: str, checkpoint_path: str, device: str,
                 checkpoint_key: str = None, strip_prefix: str = None) -> nn.Module:
    """
    Load a timm ViT and restore pretrained weights from a LeJEPA checkpoint.
    
    LeJEPA checkpoints (PyTorch Lightning) typically store the state dict under
    'state_dict' with keys prefixed by 'encoder.' or similar.
    """
    # Create model with no pretrained weights (we load our own)
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    # num_classes=0 removes the classification head, giving us the backbone
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # --- Extract state dict from checkpoint ---
    state_dict = None
    if checkpoint_key is not None:
        state_dict = ckpt[checkpoint_key]
    else:
        # Try common keys
        for key in ["state_dict", "model", "encoder", "model_state_dict"]:
            if key in ckpt:
                state_dict = ckpt[key]
                print(f"  Found state_dict under key '{key}'")
                break
        if state_dict is None:
            # Assume the checkpoint IS the state dict
            state_dict = ckpt
            print("  Using checkpoint directly as state_dict")
    
    # --- Auto-detect prefix if not specified ---
    if strip_prefix is None:
        # Inspect keys to guess prefix
        sample_keys = list(state_dict.keys())[:5]
        print(f"  Sample checkpoint keys: {sample_keys}")
        
        # Common prefixes in LeJEPA / PyTorch Lightning checkpoints
        for prefix_candidate in ["encoder.", "backbone.", "module.", "model.encoder."]:
            if any(k.startswith(prefix_candidate) for k in state_dict.keys()):
                strip_prefix = prefix_candidate
                print(f"  Auto-detected prefix to strip: '{strip_prefix}'")
                break
    
    # --- Strip prefix ---
    if strip_prefix:
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith(strip_prefix):
                new_sd[k[len(strip_prefix):]] = v
        state_dict = new_sd
    
    # --- Filter to only keys that exist in the model ---
    model_keys = set(model.state_dict().keys())
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered_sd.keys())
    unexpected = set(filtered_sd.keys()) - model_keys
    
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (showing first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (showing first 5): {list(unexpected)[:5]}")
    
    msg = model.load_state_dict(filtered_sd, strict=False)
    print(f"  load_state_dict result: {msg}")
    
    model = model.to(device).eval()
    return model


# ──────────────────────────────────────────────────────────────
# 2. Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_patch_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Extract last-layer patch token features from a timm ViT.
    
    Args:
        model: timm ViT with num_classes=0
        images: (B, 3, H, W) tensor
    
    Returns:
        patch_tokens: (B, N_patches, D) tensor
        
    For ViT-Large/14 with 224x224 input:
        N_patches = (224/14)^2 = 256
        D = 1024
    """
    # timm ViTs with num_classes=0 return (B, D) by default (pooled).
    # We need the intermediate patch tokens. Use forward_features + 
    # extract before the final pooling.
    
    # Method: use forward_features which returns (B, N+1, D) for ViTs with CLS token
    # or (B, N, D) without CLS. We then strip the CLS token.
    
    features = model.forward_features(images)
    
    # features shape depends on model:
    # - Standard ViT with CLS: (B, 1 + N_patches, D)  — CLS is index 0
    # - ViT without CLS (rare): (B, N_patches, D)
    
    if features.dim() == 2:
        raise ValueError(
            f"Got 2D output {features.shape} — model may be pooling internally. "
            "Make sure num_classes=0 and the model returns patch tokens."
        )
    
    # Check if model has CLS token
    has_cls = hasattr(model, 'cls_token') and model.cls_token is not None
    
    if has_cls:
        # Strip CLS token (index 0)
        patch_tokens = features[:, 1:, :]  # (B, N_patches, D)
    else:
        patch_tokens = features  # (B, N_patches, D)
    
    return patch_tokens


# ──────────────────────────────────────────────────────────────
# 3. PCA → RGB mapping (per-image, independently)
# ──────────────────────────────────────────────────────────────

def pca_to_rgb(patch_features: np.ndarray, 
               use_foreground_threshold: bool = False,
               fg_threshold: float = 0.0) -> np.ndarray:
    """
    Apply PCA to a single image's patch features and map first 3 PCs to RGB.
    
    Args:
        patch_features: (N_patches, D) numpy array
        use_foreground_threshold: if True, use PC1 to threshold fg/bg and 
                                  run PCA again on foreground only (DINO-style)
        fg_threshold: threshold on PC1 for foreground selection
    
    Returns:
        rgb: (N_patches, 3) numpy array, values in [0, 1]
    """
    N = patch_features.shape[0]
    
    if use_foreground_threshold:
        # Step 1: Run PCA with 1 component to get foreground mask
        pca1 = PCA(n_components=1)
        pc1 = pca1.fit_transform(patch_features).squeeze()  # (N,)
        
        # Foreground = patches where PC1 > threshold
        # (In DINO, the background tends to cluster together on one side of PC1)
        fg_mask = pc1 > fg_threshold
        
        if fg_mask.sum() < 4:
            # Fallback: not enough foreground patches, use all
            fg_mask = np.ones(N, dtype=bool)
        
        # Step 2: Run PCA with 3 components on foreground patches only
        pca3 = PCA(n_components=3)
        pca3.fit(patch_features[fg_mask])
        
        # Project ALL patches using the foreground-fitted PCA
        components = pca3.transform(patch_features)  # (N, 3)
        
        # Set background patches to a neutral color (gray)
        rgb = np.zeros((N, 3))
        for c in range(3):
            channel = components[:, c]
            fg_vals = channel[fg_mask]
            # Normalize foreground values to [0, 1]
            vmin, vmax = fg_vals.min(), fg_vals.max()
            if vmax - vmin > 1e-8:
                channel_norm = (channel - vmin) / (vmax - vmin)
            else:
                channel_norm = np.full_like(channel, 0.5)
            rgb[:, c] = channel_norm
        
        # Set background to gray
        rgb[~fg_mask] = 0.5
        
    else:
        # Simple version: PCA on all patches, map first 3 PCs to RGB
        pca = PCA(n_components=3)
        components = pca.fit_transform(patch_features)  # (N, 3)
        
        # Min-max normalize each component independently to [0, 1]
        rgb = np.zeros_like(components)
        for c in range(3):
            vmin = components[:, c].min()
            vmax = components[:, c].max()
            if vmax - vmin > 1e-8:
                rgb[:, c] = (components[:, c] - vmin) / (vmax - vmin)
            else:
                rgb[:, c] = 0.5
    
    return np.clip(rgb, 0, 1)


# ──────────────────────────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────────────────────────

def visualize_pca_features(
    original_images: list,  # list of PIL Images
    pca_maps: list,         # list of (H_grid, W_grid, 3) numpy arrays
    output_path: str,
    img_size: int = 224,
    interpolation: str = "nearest",
    ncols: int = 4,
):
    """
    Create a figure with original images alongside their PCA feature maps.
    
    Layout: Each image gets two panels (original | PCA map), arranged in a grid.
    """
    n = len(original_images)
    if n == 0:
        raise ValueError("Cannot visualize 0 images")
    nrows = max(1, math.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 5, nrows * 2.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    
    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        
        ax_orig = axes[row, col * 2]
        ax_pca = axes[row, col * 2 + 1]
        
        # Original image
        orig = original_images[idx].resize((img_size, img_size))
        ax_orig.imshow(orig)
        ax_orig.set_title("Original", fontsize=8)
        ax_orig.axis("off")
        
        # PCA map — upsample to image resolution for display
        pca_map = pca_maps[idx]  # (H_grid, W_grid, 3)
        
        if interpolation == "bilinear":
            # Use PIL for smooth upscaling
            pca_pil = Image.fromarray((pca_map * 255).astype(np.uint8))
            pca_pil = pca_pil.resize((img_size, img_size), Image.BILINEAR)
            ax_pca.imshow(pca_pil)
        else:
            # Nearest neighbor — preserves patch boundaries (more "honest")
            ax_pca.imshow(pca_map, interpolation="nearest",
                         extent=[0, img_size, img_size, 0])
        
        ax_pca.set_title("PCA Features (PC1→R, PC2→G, PC3→B)", fontsize=7)
        ax_pca.axis("off")
    
    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Collect image paths ---
    if args.image_paths:
        image_paths = args.image_paths
    elif args.image_dir:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths = sorted(image_paths)
    else:
        raise ValueError("Must provide either --image_dir or --image_paths")
    
    image_paths = image_paths[:args.num_images]
    if not image_paths:
        raise SystemExit(
            f"No images found. Check that --image_dir or --image_paths points to "
            f"valid .jpg/.jpeg/.png files. Searched: {args.image_dir or args.image_paths}"
        )
    print(f"Visualizing {len(image_paths)} images")
    
    # --- Image transforms (standard ImageNet eval preprocessing) ---
    transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # --- Load model ---
    print(f"Loading {args.arch} from {args.checkpoint}")
    model = load_encoder(
        args.arch, args.checkpoint, args.device,
        checkpoint_key=args.checkpoint_key,
        strip_prefix=args.strip_prefix,
    )
    
    # Compute spatial grid dimensions
    grid_h = args.img_size // args.patch_size
    grid_w = args.img_size // args.patch_size
    print(f"Patch grid: {grid_h} x {grid_w} = {grid_h * grid_w} patches")
    
    # --- Process images ---
    original_images = []
    pca_maps = []
    
    for img_path in image_paths:
        print(f"  Processing: {os.path.basename(img_path)}")
        
        # Load and preprocess
        pil_img = Image.open(img_path).convert("RGB")
        original_images.append(pil_img)
        
        img_tensor = transform(pil_img).unsqueeze(0).to(args.device)  # (1, 3, H, W)
        
        # Extract patch features
        patch_tokens = extract_patch_features(model, img_tensor)  # (1, N, D)
        patch_features = patch_tokens[0].cpu().numpy()  # (N, D)
        
        # Sanity check
        expected_n = grid_h * grid_w
        assert patch_features.shape[0] == expected_n, \
            f"Expected {expected_n} patches, got {patch_features.shape[0]}"
        
        # PCA → RGB
        rgb = pca_to_rgb(
            patch_features,
            use_foreground_threshold=args.use_foreground_threshold,
            fg_threshold=args.fg_threshold,
        )  # (N, 3)
        
        # Reshape to spatial grid
        pca_map = rgb.reshape(grid_h, grid_w, 3)  # (H_grid, W_grid, 3)
        pca_maps.append(pca_map)
    
    # --- Visualize ---
    output_path = os.path.join(args.output_dir, "pca_features.png")
    visualize_pca_features(
        original_images, pca_maps, output_path,
        img_size=args.img_size,
        interpolation=args.interpolation,
    )
    
    # Also save individual PCA maps as standalone images
    for idx, (pca_map, img_path) in enumerate(zip(pca_maps, image_paths)):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save raw PCA map (low res, nearest-neighbor is faithful)
        pca_img = Image.fromarray((pca_map * 255).astype(np.uint8))
        pca_img_upscaled = pca_img.resize(
            (args.img_size, args.img_size),
            Image.NEAREST if args.interpolation == "nearest" else Image.BILINEAR
        )
        pca_img_upscaled.save(os.path.join(args.output_dir, f"{basename}_pca.png"))
    
    print("Done!")


if __name__ == "__main__":
    main()