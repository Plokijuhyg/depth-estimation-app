"""
train.py — Monocular Depth Estimation Training Pipeline

Usage
-----
  # Full training (GPU recommended):
  python train.py --data dataset/ --epochs 20 --batch-size 8

  # Quick CPU test (finishes in ~5 minutes):
  python train.py --data dataset/ --epochs 2 --batch-size 1 --img-h 128 --img-w 160 --max-batches 50
"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from dataset_loader import get_dataloaders
from model          import DepthEstimationNet
from loss           import DepthLoss
from utils          import (
    compute_depth_metrics,
    MetricTracker,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
)


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train Monocular Depth Estimation")
    p.add_argument("--data",        type=str,   default="dataset",
                   help="Dataset root folder (contains nyu2_train.csv / nyu2_train/)")
    p.add_argument("--backbone",    type=str,   default="resnet34",
                   choices=["resnet18", "resnet34"])
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--img-h",       type=int,   default=240)
    p.add_argument("--img-w",       type=int,   default=320)
    p.add_argument("--out-dir",     type=str,   default="checkpoints")
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--workers",     type=int,   default=0,
                   help="Use 0 on Windows")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--alpha",       type=float, default=1.0)
    p.add_argument("--beta",        type=float, default=1.0)
    p.add_argument("--gamma",       type=float, default=0.5)
    p.add_argument("--max-batches", type=int,   default=None,
                   help="Limit batches per epoch (CPU test mode, e.g. --max-batches 50)")
    return p.parse_args()


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, device, epoch, total_epochs, max_batches=None):
    model.train()
    loss_meter = MetricTracker(
        ["loss/total", "loss/l1", "loss/silog", "loss/gradient"]
    )
    t0    = time.time()
    n_bat = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_idx, (images, depths) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)

        pred          = model(images)
        loss, details = criterion(pred, depths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_meter.update(details, n=images.size(0))

        # Print progress every 20% of batches
        if (batch_idx + 1) % max(1, n_bat // 5) == 0 or batch_idx == n_bat - 1:
            elapsed = time.time() - t0
            eta     = elapsed / (batch_idx + 1) * (n_bat - batch_idx - 1)
            print(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx+1}/{n_bat}] "
                f"Loss: {loss_meter.avg('loss/total'):.4f} "
                f"Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s"
            )

    return loss_meter.summary()


# ─────────────────────────────────────────────
# Validate one epoch
# ─────────────────────────────────────────────

def validate(model, loader, criterion, device, max_batches=None):
    model.eval()
    metric_keys  = ["RMSE", "MAE", "AbsRel", "SqRel", "RMSE_log",
                    "delta1", "delta2", "delta3"]
    loss_keys    = ["loss/total", "loss/l1", "loss/silog", "loss/gradient"]
    metric_meter = MetricTracker(metric_keys)
    loss_meter   = MetricTracker(loss_keys)

    with torch.no_grad():
        for batch_idx, (images, depths) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            pred          = model(images)
            loss, details = criterion(pred, depths)
            metrics       = compute_depth_metrics(pred, depths)

            b = images.size(0)
            loss_meter.update(details, n=b)
            metric_meter.update(metrics, n=b)

    results = {}
    results.update(loss_meter.summary())
    results.update(metric_meter.summary())
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warn if CPU and no --max-batches set
    if device.type == "cpu" and args.max_batches is None:
        print("\n  WARNING: Training on CPU without --max-batches will take many hours.")
        print("  Recommended for CPU testing:")
        print("    python train.py --data dataset/ --batch-size 1 --img-h 128 --img-w 160 --epochs 2 --max-batches 50\n")

    print("\n" + "=" * 60)
    print("  Monocular Depth Estimation — Training")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Data        : {args.data}")
    print(f"  Backbone    : {args.backbone}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Image size  : {args.img_h} x {args.img_w}")
    print(f"  Max batches : {args.max_batches or 'all'}")
    print("=" * 60 + "\n")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data_root   = args.data,
        batch_size  = args.batch_size,
        img_size    = (args.img_h, args.img_w),
        num_workers = args.workers,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = DepthEstimationNet(
        backbone   = args.backbone,
        pretrained = not args.no_pretrain,
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"  Parameters: {total:,} total  |  {trainable:,} trainable\n")

    # ── Loss ──────────────────────────────────────────────────────
    criterion = DepthLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    # ── Optimiser (encoder LR 10x lower — it's already pretrained) ─
    encoder_params = (
        list(model.enc0.parameters()) +
        list(model.enc1.parameters()) +
        list(model.enc2.parameters()) +
        list(model.enc3.parameters()) +
        list(model.enc4.parameters())
    )
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": decoder_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    # ── Scheduler ─────────────────────────────────────────────────
    batches_per_epoch = (
        min(len(train_loader), args.max_batches)
        if args.max_batches else len(train_loader)
    )
    total_steps = args.epochs * batches_per_epoch
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = [args.lr * 0.1, args.lr],
        total_steps     = total_steps,
        pct_start       = 0.3,
        anneal_strategy = "cos",
    )

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_rmse   = float("inf")
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_rmse = \
            load_checkpoint(args.resume, model, optimizer, scheduler)

    # ── Log ───────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,RMSE,MAE,AbsRel,delta1\n")

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'─'*60}")
        print(f"  EPOCH {epoch}/{args.epochs}   LR={optimizer.param_groups[1]['lr']:.2e}")
        print(f"{'─'*60}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, args.epochs, args.max_batches,
        )

        print("\n  Validating...")
        val_metrics = validate(
            model, val_loader, criterion, device,
            max_batches=min(20, len(val_loader)),   # cap val at 20 batches on CPU
        )
        print_metrics(val_metrics, prefix="Validation")

        # Log
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_metrics.get('loss/total', 0):.5f},"
                f"{val_metrics.get('loss/total', 0):.5f},"
                f"{val_metrics.get('RMSE', 0):.5f},"
                f"{val_metrics.get('MAE', 0):.5f},"
                f"{val_metrics.get('AbsRel', 0):.5f},"
                f"{val_metrics.get('delta1', 0):.5f}\n"
            )

        # Checkpoint
        val_rmse = val_metrics.get("RMSE", float("inf"))
        is_best  = val_rmse < best_rmse
        if is_best:
            best_rmse = val_rmse

        save_checkpoint(
            state = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_metric":     best_rmse,
                "args":            vars(args),
            },
            checkpoint_dir = args.out_dir,
            filename       = f"checkpoint_epoch{epoch:03d}.pth",
            is_best        = is_best,
        )

        # Remove old checkpoints (keep last 3)
        for old in range(1, epoch - 2):
            old_path = os.path.join(args.out_dir, f"checkpoint_epoch{old:03d}.pth")
            if os.path.exists(old_path):
                os.remove(old_path)

    print("\n" + "=" * 60)
    print(f"  Done!  Best RMSE: {best_rmse:.4f} m")
    print(f"  Best model  → {args.out_dir}/best_model.pth")
    print(f"  Training log → {log_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()