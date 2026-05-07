"""
utils.py
--------
Evaluation metrics, checkpoint helpers, and training utilities
for Monocular Depth Estimation.

Metrics implemented
-------------------
All metrics are computed only over valid (> MIN_DEPTH) pixels.

1. RMSE    – Root Mean Squared Error  (lower is better)
             √( (1/n) Σ (pred - gt)² )

2. MAE     – Mean Absolute Error  (lower is better)
             (1/n) Σ |pred - gt|

3. AbsRel  – Absolute Relative Error  (lower is better)
             (1/n) Σ |pred - gt| / gt

4. SqRel   – Squared Relative Error  (lower is better)
             (1/n) Σ (pred - gt)² / gt

5. δ < 1.25   (higher is better) – % of pixels where
   δ < 1.25²  max(pred/gt, gt/pred) < threshold
   δ < 1.25³
"""

import os
import torch
import numpy as np


MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_depth_metrics(pred: torch.Tensor,
                           gt:   torch.Tensor) -> dict:
    """
    Compute standard monocular depth evaluation metrics.

    Parameters
    ----------
    pred : Tensor (B, 1, H, W) or (B, H, W)  predicted depth
    gt   : Tensor (B, 1, H, W) or (B, H, W)  ground-truth depth

    Returns
    -------
    dict of metric names → float values
    """
    # Flatten spatial dims
    pred = pred.squeeze(1)   # (B, H, W)
    gt   = gt.squeeze(1)

    # Valid mask
    mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)

    pred_v = pred[mask].float()
    gt_v   = gt[mask].float()

    # Clamp predictions to valid range
    pred_v = torch.clamp(pred_v, MIN_DEPTH, MAX_DEPTH)

    # ── Error metrics ──────────────────────────────────────────────
    diff      = pred_v - gt_v
    abs_diff  = torch.abs(diff)

    rmse    = torch.sqrt((diff ** 2).mean()).item()
    mae     = abs_diff.mean().item()
    abs_rel = (abs_diff / gt_v).mean().item()
    sq_rel  = ((diff ** 2) / gt_v).mean().item()

    # Log scale RMSE
    log_diff  = torch.log(pred_v) - torch.log(gt_v)
    rmse_log  = torch.sqrt((log_diff ** 2).mean()).item()

    # ── Threshold accuracy ─────────────────────────────────────────
    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)

    delta1 = (ratio < 1.25    ).float().mean().item()
    delta2 = (ratio < 1.25**2 ).float().mean().item()
    delta3 = (ratio < 1.25**3 ).float().mean().item()

    return {
        "RMSE":     rmse,
        "MAE":      mae,
        "AbsRel":   abs_rel,
        "SqRel":    sq_rel,
        "RMSE_log": rmse_log,
        "delta1":   delta1,
        "delta2":   delta2,
        "delta3":   delta3,
    }


class AverageMeter:
    """Running average of a scalar value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class MetricTracker:
    """Track multiple AverageMeters simultaneously."""

    def __init__(self, keys):
        self.meters = {k: AverageMeter(k) for k in keys}

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def update(self, metrics_dict: dict, n: int = 1):
        for k, v in metrics_dict.items():
            if k in self.meters:
                self.meters[k].update(v, n)

    def avg(self, key: str) -> float:
        return self.meters[key].avg

    def summary(self) -> dict:
        return {k: m.avg for k, m in self.meters.items()}


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"  ★ New best model saved → {best_path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint. Returns (model, optimizer, scheduler, start_epoch, best_metric).
    """
    assert os.path.isfile(path), f"Checkpoint not found: {path}"
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])
    start_epoch  = ckpt.get("epoch", 0) + 1
    best_metric  = ckpt.get("best_metric", float("inf"))

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    print(f"  Loaded checkpoint '{path}' (epoch {start_epoch-1})")
    return model, optimizer, scheduler, start_epoch, best_metric


# ─────────────────────────────────────────────
# Pretty metric printer
# ─────────────────────────────────────────────

def print_metrics(metrics: dict, prefix: str = ""):
    header = f"{prefix} Metrics" if prefix else "Metrics"
    bar    = "─" * 50
    print(f"\n{bar}")
    print(f"  {header}")
    print(bar)
    error_keys  = ["RMSE", "MAE", "AbsRel", "SqRel", "RMSE_log"]
    thresh_keys = ["delta1", "delta2", "delta3"]

    print("  Error metrics (lower ↓ is better):")
    for k in error_keys:
        if k in metrics:
            print(f"    {k:<12} : {metrics[k]:.4f}")

    print("  Threshold accuracy (higher ↑ is better):")
    labels = {"delta1": "δ < 1.25", "delta2": "δ < 1.25²", "delta3": "δ < 1.25³"}
    for k in thresh_keys:
        if k in metrics:
            print(f"    {labels[k]:<12} : {metrics[k]*100:.2f} %")
    print(bar)


# ─────────────────────────────────────────────
# Depth → colourmap (for visualisation)
# ─────────────────────────────────────────────

def depth_to_colormap(depth_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, H, W) or (H, W) depth tensor to an RGB colormap array (H, W, 3).
    Uses 'plasma' colormap style via numpy.
    """
    import matplotlib.cm as cm

    d = depth_tensor.squeeze().cpu().numpy()
    d_norm = np.clip((d - d.min()) / (d.max() - d.min() + 1e-8), 0.0, 1.0)
    colormap = cm.plasma(d_norm)[:, :, :3]   # (H, W, 3) float32 [0,1]
    return (colormap * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pred = torch.rand(4, 1, 240, 320) * 10.0
    gt   = torch.rand(4, 1, 240, 320) * 10.0

    metrics = compute_depth_metrics(pred, gt)
    print_metrics(metrics, prefix="Random Baseline")
