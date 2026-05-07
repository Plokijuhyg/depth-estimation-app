"""
visualize.py
------------
Standalone visualisation utilities for Monocular Depth Estimation.

Functions
---------
plot_sample            – Show RGB | GT depth | Predicted depth side-by-side
plot_error_map         – Show absolute error heat-map
plot_training_curves   – Plot train/val loss curves from CSV log
plot_depth_histogram   – Histogram of GT vs predicted depths
create_depth_overlay   – Overlay colourised depth on RGB (blended)
save_all_visuals       – Run every plot for a set of samples

Standalone usage
----------------
    python visualize.py \
        --checkpoint checkpoints/best_model.pth \
        --data       dataset/nyu_depth_v2_labeled.mat \
        --log        checkpoints/train_log.csv \
        --out-dir    results/visualisations
"""

import os
import argparse

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image

from model          import DepthEstimationNet, MAX_DEPTH
from utils          import compute_depth_metrics, depth_to_colormap
from dataset_loader import NYU_MEAN, NYU_STD, get_dataloaders


# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────
CMAP = "plasma"


# ─────────────────────────────────────────────
# 1. Side-by-side plot: RGB | GT | Predicted
# ─────────────────────────────────────────────

def plot_sample(rgb_np: np.ndarray,
                gt_np:  np.ndarray,
                pred_np: np.ndarray,
                metrics: dict = None,
                title:   str  = "",
                save_path: str = None):
    """
    Parameters
    ----------
    rgb_np   : (H, W, 3)  float32 [0, 1]
    gt_np    : (H, W)     float32 metres
    pred_np  : (H, W)     float32 metres
    """
    # Use shared vmin/vmax based on actual data range so both maps
    # use the same colour scale and are visually comparable
    valid = gt_np[gt_np > 0.001]
    vmin  = float(valid.min()) if len(valid) else 0.0
    vmax  = float(valid.max()) if len(valid) else MAX_DEPTH

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    # ── RGB ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.clip(rgb_np, 0, 1))
    ax0.set_title("Input RGB Image", fontsize=11, fontweight="bold")
    ax0.axis("off")

    # ── Ground truth ──────────────────────────
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(gt_np, cmap=CMAP + "_r", vmin=vmin, vmax=vmax)
    ax1.set_title("Ground-Truth Depth", fontsize=11, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Depth (m)")

    # ── Prediction ────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(pred_np, cmap=CMAP , vmin=vmin, vmax=vmax)
    ax2.set_title("Predicted Depth", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth (m)")

    # ── Metrics subtitle ──────────────────────
    if metrics:
        subtitle = (
            f"RMSE={metrics.get('RMSE', 0):.4f}m  "
            f"MAE={metrics.get('MAE', 0):.4f}m  "
            f"AbsRel={metrics.get('AbsRel', 0):.4f}  "
            f"δ<1.25={metrics.get('delta1', 0)*100:.1f}%"
        )
        fig.suptitle(subtitle, fontsize=9, y=0.01,
                     style="italic", color="#555555")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────
# 2. Absolute error heat-map
# ─────────────────────────────────────────────

def plot_error_map(rgb_np: np.ndarray,
                   gt_np:  np.ndarray,
                   pred_np: np.ndarray,
                   save_path: str = None):
    abs_err = np.abs(pred_np - gt_np)
    mask    = gt_np > 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Error Analysis", fontsize=13, fontweight="bold")

    axes[0].imshow(np.clip(rgb_np, 0, 1))
    axes[0].set_title("RGB"); axes[0].axis("off")

    err_plot = np.zeros_like(abs_err)
    err_plot[mask] = abs_err[mask]
    im = axes[1].imshow(err_plot, cmap="hot", vmin=0, vmax=1.0)
    axes[1].set_title("Absolute Error (m)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Relative error
    rel_err = np.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / (gt_np[mask] + 1e-8)
    im2 = axes[2].imshow(rel_err, cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title("Relative Error")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────
# 3. Training curves from CSV log
# ─────────────────────────────────────────────

def plot_training_curves(log_csv: str, save_path: str = None):
    import csv

    epochs      = []
    train_loss  = []
    val_loss    = []
    rmse        = []
    delta1      = []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            rmse.append(float(row["RMSE"]))
            delta1.append(float(row["delta1"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training Progress", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color="#E74C3C", lw=2)
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="#3498DB", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # RMSE
    axes[1].plot(epochs, rmse, color="#2ECC71", lw=2, marker="o", ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (m)")
    axes[1].set_title("Validation RMSE"); axes[1].grid(alpha=0.3)

    # δ < 1.25
    axes[2].plot(epochs, [d*100 for d in delta1],
                 color="#F39C12", lw=2, marker="s", ms=3)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("δ < 1.25 (%)")
    axes[2].set_title("Threshold Accuracy δ < 1.25"); axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────
# 4. Depth distribution histogram
# ─────────────────────────────────────────────

def plot_depth_histogram(gt_np: np.ndarray,
                          pred_np: np.ndarray,
                          save_path: str = None):
    mask = gt_np > 1e-3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt_np[mask].ravel(),   bins=80, alpha=0.6,
            color="#3498DB", label="Ground Truth", density=True)
    ax.hist(pred_np[mask].ravel(), bins=80, alpha=0.6,
            color="#E74C3C", label="Predicted",    density=True)
    ax.set_xlabel("Depth (metres)")
    ax.set_ylabel("Density")
    ax.set_title("Depth Distribution: GT vs Predicted")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────
# 5. Depth overlay on RGB
# ─────────────────────────────────────────────

def create_depth_overlay(rgb_np:   np.ndarray,
                          depth_np: np.ndarray,
                          alpha: float = 0.5,
                          save_path: str = None):
    """
    Blend colourised depth on top of the RGB image.
    rgb_np   : (H,W,3) float32 [0,1]
    depth_np : (H,W)   float32 metres
    """
    d_norm     = np.clip((depth_np - depth_np.min()) /
                         (depth_np.max() - depth_np.min() + 1e-8), 0, 1)
    depth_col  = cm.plasma(d_norm)[:, :, :3]   # (H,W,3) [0,1]

    overlay = alpha * depth_col + (1 - alpha) * np.clip(rgb_np, 0, 1)
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Depth Overlay", fontsize=12, fontweight="bold")

    axes[0].imshow(np.clip(rgb_np, 0, 1))
    axes[0].set_title("RGB"); axes[0].axis("off")

    axes[1].imshow(depth_col)
    axes[1].set_title("Depth Colormap"); axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Blended (α={alpha})"); axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────
# 6. Batch runner
# ─────────────────────────────────────────────

def save_all_visuals(model, val_loader, device, out_dir: str, n_samples: int = 8):
    """Run all visualisation types for the first n_samples from val_loader."""
    os.makedirs(out_dir, exist_ok=True)

    inv_mean = torch.tensor(NYU_MEAN).view(1, 3, 1, 1)
    inv_std  = torch.tensor(NYU_STD ).view(1, 3, 1, 1)

    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for images, depths in val_loader:
            if sample_idx >= n_samples:
                break

            preds = model(images.to(device)).cpu()

            # Denorm images
            imgs_vis = (images * inv_std + inv_mean).clamp(0, 1)

            for i in range(images.size(0)):
                if sample_idx >= n_samples:
                    break

                rgb_np  = imgs_vis[i].permute(1, 2, 0).numpy()
                gt_np   = depths[i, 0].numpy()
                pred_np = preds[i, 0].numpy()

                # Individual metrics for this sample
                m = compute_depth_metrics(preds[i:i+1], depths[i:i+1])

                base = os.path.join(out_dir, f"sample_{sample_idx+1:03d}")

                plot_sample(
                    rgb_np, gt_np, pred_np,
                    metrics   = m,
                    title     = f"Sample {sample_idx+1}",
                    save_path = f"{base}_comparison.png",
                )
                plot_error_map(rgb_np, gt_np, pred_np,
                               save_path=f"{base}_error.png")
                create_depth_overlay(rgb_np, pred_np,
                                     save_path=f"{base}_overlay.png")
                if sample_idx == 0:
                    plot_depth_histogram(gt_np, pred_np,
                                         save_path=os.path.join(out_dir, "depth_histogram.png"))

                sample_idx += 1

    print(f"\n  All visualisations saved to: {out_dir}/")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Visualise depth estimation results")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data",       type=str,
                   default="dataset/nyu_depth_v2_labeled.mat")
    p.add_argument("--log",        type=str, default=None,
                   help="Path to train_log.csv (optional)")
    p.add_argument("--backbone",   type=str, default="resnet34",
                   choices=["resnet18", "resnet34"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-h",      type=int, default=240)
    p.add_argument("--img-w",      type=int, default=320)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--out-dir",    type=str, default="results/visualisations")
    p.add_argument("--n-samples",  type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from utils import load_checkpoint
    model = DepthEstimationNet(backbone=args.backbone, pretrained=False)
    model, _, _, _, _ = load_checkpoint(args.checkpoint, model)
    model = model.to(device)

    # Val loader
    _, val_loader = get_dataloaders(
        hdf5_path   = args.data,
        batch_size  = args.batch_size,
        img_size    = (args.img_h, args.img_w),
        num_workers = args.workers,
    )

    save_all_visuals(model, val_loader, device, args.out_dir, args.n_samples)

    # Training curves (optional)
    if args.log and os.path.exists(args.log):
        plot_training_curves(
            args.log,
            save_path=os.path.join(args.out_dir, "training_curves.png"),
        )