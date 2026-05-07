"""
video.py
--------
Monocular Depth Estimation on Video
=====================================
بيشتغل على فيديو frame بـ frame ويطلع فيديو فيه:
    - الـ RGB الأصلي
    - الـ depth map بالألوان

Usage
-----
    python video.py --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
    python video.py --checkpoint checkpoints/best_model.pth --video video.mp4 --out results/output.mp4
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.cm as cm
from PIL import Image

from model          import DepthEstimationNet, MAX_DEPTH
from dataset_loader import NYU_MEAN, NYU_STD


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Depth Estimation on Video")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model .pth")
    p.add_argument("--video",      type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--out",        type=str, default=None,
                   help="Output video path (default: results/<input>_depth.mp4)")
    p.add_argument("--backbone",   type=str, default="resnet34",
                   choices=["resnet18", "resnet34"])
    p.add_argument("--img-h",      type=int, default=240)
    p.add_argument("--img-w",      type=int, default=320)
    p.add_argument("--side-by-side", action="store_true", default=True,
                   help="Show RGB and depth side by side (default: True)")
    return p.parse_args()


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray, img_size: tuple) -> torch.Tensor:
    """
    OpenCV frame (BGR, uint8) → normalised tensor (1, 3, H, W)
    """
    H, W    = img_size
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((W, H), Image.BILINEAR)
    tensor  = TF.to_tensor(pil_img)
    tensor  = T.Normalize(mean=NYU_MEAN, std=NYU_STD)(tensor)
    return tensor.unsqueeze(0)   # (1, 3, H, W)


# ─────────────────────────────────────────────
# Depth → colour
# ─────────────────────────────────────────────

def depth_to_color(depth_np: np.ndarray, out_size: tuple) -> np.ndarray:
    """
    (H, W) float32 metres → (H, W, 3) uint8 BGR colourmap
    """
    d_norm   = np.clip((depth_np - depth_np.min()) /
                       (depth_np.max() - depth_np.min() + 1e-8), 0, 1)
    colored  = (cm.plasma(d_norm)[:, :, :3] * 255).astype(np.uint8)
    colored  = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    H, W     = out_size
    colored  = cv2.resize(colored, (W, H), interpolation=cv2.INTER_LINEAR)
    return colored


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output path ───────────────────────────────────────────────
    if args.out is None:
        os.makedirs("results", exist_ok=True)
        base     = os.path.splitext(os.path.basename(args.video))[0]
        args.out = os.path.join("results", f"{base}_depth.mp4")

    print(f"\n  Device     : {device}")
    print(f"  Video in   : {args.video}")
    print(f"  Video out  : {args.out}")

    # ── Load model ────────────────────────────────────────────────
    model = DepthEstimationNet(backbone=args.backbone, pretrained=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"  Model loaded: {args.checkpoint}\n")

    # ── Open video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output frame size
    out_w = orig_w * 2   # side by side
    out_h = orig_h

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    print(f"  Input  : {orig_w}×{orig_h}  {fps:.1f}fps  {total} frames")
    print(f"  Output : {out_w}×{out_h} (side-by-side)\n")

    img_size   = (args.img_h, args.img_w)
    frame_idx  = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # ── Preprocess ────────────────────────────────────────
            tensor = preprocess_frame(frame, img_size).to(device)

            # ── Predict ───────────────────────────────────────────
            depth_pred = model(tensor)                          # (1,1,H,W)
            depth_np   = depth_pred.squeeze().cpu().numpy()    # (H,W)

            # ── Colourize depth ───────────────────────────────────
            depth_col  = depth_to_color(depth_np, (orig_h, orig_w))

            # ── Add labels ────────────────────────────────────────
            cv2.putText(frame,     "RGB Input",    (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(depth_col, "Predicted Depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # ── Combine side by side ──────────────────────────────
            combined = np.hstack([frame, depth_col])
            writer.write(combined)

            # Progress
            if frame_idx % 30 == 0 or frame_idx == total:
                print(f"  Frame {frame_idx}/{total}  ({frame_idx/total*100:.1f}%)")

    cap.release()
    writer.release()

    print(f"\n  Done! Saved → {args.out}\n")
 

if __name__ == "__main__":
    main()