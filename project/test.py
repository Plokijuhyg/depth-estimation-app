"""
test.py
-------
Testing & Inference Script for Monocular Depth Estimation
==========================================================

Two modes:
    1. Evaluate on the full validation split (with metrics)
    2. Predict depth for a single image file

Usage
-----
# Mode 1: evaluate on dataset
    python test.py --checkpoint checkpoints/best_model.pth \
                   --data dataset/nyu_depth_v2_labeled.mat

# Mode 2: predict on a single image
    python test.py --checkpoint checkpoints/best_model.pth \
                   --image path/to/photo.jpg
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; remove if running interactively
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model  import DepthEstimationNet, MAX_DEPTH
from utils  import compute_depth_metrics, print_metrics, MetricTracker
from dataset_loader import NYU_MEAN, NYU_STD, MIN_DEPTH, get_dataloaders


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Test / Inference: Monocular Depth")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model .pth file")
    p.add_argument("--data",       type=str,
                   default="dataset/nyu_depth_v2_labeled.mat",
                   help="Dataset file (for evaluation mode)")
    p.add_argument("--image",      type=str, default=None,
                   help="Path to a single image for inference")
    p.add_argument("--backbone",   type=str, default="resnet34",
                   choices=["resnet18", "resnet34"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-h",      type=int, default=240)
    p.add_argument("--img-w",      type=int, default=320)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--out-dir",    type=str, default="results",
                   help="Directory to save output images")
    p.add_argument("--num-vis",    type=int, default=5,
                   help="Number of validation samples to visualise")
    return p.parse_args()


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, backbone: str, device: torch.device):
    model = DepthEstimationNet(backbone=backbone, pretrained=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both raw state-dict and full checkpoint dict
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        epoch      = ckpt.get("epoch", "?")
        print(f"  Loaded checkpoint from epoch {epoch}")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded: {checkpoint_path}\n")
    return model


# ─────────────────────────────────────────────
# Pre-process a single image
# ─────────────────────────────────────────────

def preprocess_image(path: str, img_size: tuple) -> torch.Tensor:
    """Returns a (1, 3, H, W) normalised tensor ready for the model."""
    img = Image.open(path).convert("RGB")
    H, W = img_size
    img = img.resize((W, H), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = T.Normalize(mean=NYU_MEAN, std=NYU_STD)(tensor)
    return tensor.unsqueeze(0)   # (1, 3, H, W)


# ─────────────────────────────────────────────
# Single-image inference mode
# ─────────────────────────────────────────────

def infer_single_image(model, image_path: str, img_size: tuple,
                       device: torch.device, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Predicting depth for: {image_path}")

    img_tensor = preprocess_image(image_path, img_size).to(device)

    with torch.no_grad():
        depth_pred = model(img_tensor)          # (1, 1, H, W)

    # ── Postprocess ───────────────────────────────────────────────
    depth_np = depth_pred.squeeze().cpu().numpy()   # (H, W)  metres

    # Denormalise input image for display
    img_display = Image.open(image_path).convert("RGB")
    img_display = img_display.resize((img_size[1], img_size[0]), Image.BILINEAR)
    img_display = np.array(img_display)

    # Colourmap
    d_norm    = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_col = (cm.plasma(d_norm)[:, :, :3] * 255).astype(np.uint8)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Monocular Depth Estimation", fontsize=14, fontweight="bold")

    axes[0].imshow(img_display)
    axes[0].set_title("Input RGB Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_np, cmap="plasma",
                        vmin=depth_np.min(), vmax=depth_np.max())
    axes[1].set_title("Predicted Depth Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (metres)")

    plt.tight_layout()
    base  = os.path.splitext(os.path.basename(image_path))[0]
    save  = os.path.join(out_dir, f"{base}_depth.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved depth visualisation → {save}")
    print(f"  Depth range: [{depth_np.min():.3f}, {depth_np.max():.3f}] metres")


# ─────────────────────────────────────────────
# Evaluation on dataset
# ─────────────────────────────────────────────

def evaluate(model, val_loader, device, out_dir: str, num_vis: int):
    os.makedirs(out_dir, exist_ok=True)

    metric_keys  = ["RMSE", "MAE", "AbsRel", "SqRel",
                    "RMSE_log", "delta1", "delta2", "delta3"]
    metric_meter = MetricTracker(metric_keys)

    vis_count    = 0
    inv_norm_mean = torch.tensor(NYU_MEAN).view(1, 3, 1, 1)
    inv_norm_std  = torch.tensor(NYU_STD ).view(1, 3, 1, 1)

    print("  Evaluating on validation set...")

    with torch.no_grad():
        for batch_idx, (images, depths) in enumerate(val_loader):
            images_dev = images.to(device, non_blocking=True)
            depths_dev = depths.to(device, non_blocking=True)

            pred_depths = model(images_dev)

            metrics = compute_depth_metrics(pred_depths, depths_dev)
            metric_meter.update(metrics, n=images.size(0))

            # ── Visualise first num_vis samples ───────────────────
            if vis_count < num_vis:
                images_np = (images * inv_norm_std + inv_norm_mean)
                images_np = images_np.clamp(0, 1).permute(0, 2, 3, 1).numpy()

                for i in range(min(images.size(0), num_vis - vis_count)):
                    rgb_np   = images_np[i]                           # (H,W,3)
                    gt_np    = depths[i, 0].numpy()                   # (H,W)
                    pred_np  = pred_depths[i, 0].cpu().numpy()        # (H,W)

                    _save_triple(rgb_np, gt_np, pred_np,
                                 out_dir, idx=vis_count)
                    vis_count += 1

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{len(val_loader)} done")

    final = metric_meter.summary()
    print_metrics(final, prefix="Test / Validation")
    return final


def _save_triple(rgb_np, gt_np, pred_np, out_dir, idx):
    """Save a side-by-side figure: RGB | GT depth | Predicted depth."""
    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sample {idx+1}", fontsize=12, fontweight="bold")

    axes[0].imshow(np.clip(rgb_np, 0, 1))
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground-Truth Depth")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="m")

    im2 = axes[2].imshow(pred_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[2].set_title("Predicted Depth")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="m")

    plt.tight_layout()
    path = os.path.join(out_dir, f"sample_{idx+1:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Visualisation saved → {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "═" * 60)
    print("  Monocular Depth Estimation — Testing")
    print("═" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print("═" * 60 + "\n")

    model = load_model(args.checkpoint, args.backbone, device)

    # ── Mode 2: single image ──────────────────────────────────────
    if args.image is not None:
        infer_single_image(
            model, args.image,
            img_size = (args.img_h, args.img_w),
            device   = device,
            out_dir  = args.out_dir,
        )
        return

    # ── Mode 1: full dataset evaluation ──────────────────────────
    if not os.path.exists(args.data):
        print(f"  ERROR: Dataset file not found: {args.data}")
        print("  Use --image <path> for single-image inference.")
        sys.exit(1)

    _, val_loader = get_dataloaders(
        hdf5_path   = args.data,
        batch_size  = args.batch_size,
        img_size    = (args.img_h, args.img_w),
        num_workers = args.workers,
    )

    evaluate(model, val_loader, device, args.out_dir, args.num_vis)


if __name__ == "__main__":
    main()
