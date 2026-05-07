"""
reconstruct_3d.py
-----------------
3D Reconstruction from Monocular Depth Estimation
===================================================
بياخد صورة RGB، بيتنبأ بالـ depth بالموديل،
وبيعمل point cloud تفاعلي ثلاثي الأبعاد بـ Open3D.

الفكرة الرياضية
---------------
كل pixel (u, v) + depth d → نقطة 3D (X, Y, Z):
    Z = d
    X = (u - cx) × Z / fx
    Y = (v - cy) × Z / fy

Usage
-----
    # من صورة:
    python reconstruct_3d.py --checkpoint checkpoints/best_model.pth --image room.jpg

    # من dataset:
    python reconstruct_3d.py --checkpoint checkpoints/best_model.pth --data dataset/ --idx 5
"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

from model          import DepthEstimationNet, MAX_DEPTH
from dataset_loader import NYU_MEAN, NYU_STD, get_dataloaders


# ─────────────────────────────────────────────
# NYU Depth V2 Camera Intrinsics (Kinect)
# ─────────────────────────────────────────────
NYU_FX = 518.8579
NYU_FY = 519.4696
NYU_CX = 325.5824
NYU_CY = 253.7362


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="3D Reconstruction from Monocular Depth")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image",      type=str, default=None,
                   help="Path to a single RGB image")
    p.add_argument("--data",       type=str, default="dataset",
                   help="Dataset root (used if --image not given)")
    p.add_argument("--idx",        type=int, default=0,
                   help="Sample index from dataset")
    p.add_argument("--backbone",   type=str, default="resnet34",
                   choices=["resnet18", "resnet34"])
    p.add_argument("--img-h",      type=int, default=240)
    p.add_argument("--img-w",      type=int, default=320)
    p.add_argument("--out-dir",    type=str, default="results/3d")
    p.add_argument("--max-points", type=int, default=50000,
                   help="Max points for matplotlib plot (Open3D uses all points)")
    p.add_argument("--no-open3d",  action="store_true",
                   help="Skip Open3D interactive viewer")
    return p.parse_args()


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

def load_model(checkpoint_path, backbone, device):
    model = DepthEstimationNet(backbone=backbone, pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"  Model loaded: {checkpoint_path}")
    return model


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────

def preprocess(img_pil, img_size):
    H, W    = img_size
    img_pil = img_pil.resize((W, H), Image.BILINEAR)
    tensor  = TF.to_tensor(img_pil)
    tensor  = T.Normalize(mean=NYU_MEAN, std=NYU_STD)(tensor)
    return tensor.unsqueeze(0)


# ─────────────────────────────────────────────
# Depth → Point Cloud
# ─────────────────────────────────────────────

def depth_to_pointcloud(depth_np, rgb_np, img_h, img_w):
    """
    بيحوّل (H,W) depth map + (H,W,3) RGB لـ point cloud.

    Returns
    -------
    points : (N, 3)  X Y Z بالـ metres
    colors : (N, 3)  R G B في [0,1]
    """
    scale_x = img_w / 640.0
    scale_y = img_h / 480.0
    fx = NYU_FX * scale_x
    fy = NYU_FY * scale_y
    cx = NYU_CX * scale_x
    cy = NYU_CY * scale_y

    u  = np.arange(img_w, dtype=np.float32)
    v  = np.arange(img_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth_np
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb_np.reshape(-1, 3)

    mask   = (Z.flatten() > 0.1) & (Z.flatten() < MAX_DEPTH)
    return points[mask], colors[mask]


# ─────────────────────────────────────────────
# Save .PLY
# ─────────────────────────────────────────────

def save_ply(points, colors, path):
    """
    بيحفظ الـ point cloud كـ .ply file.
    تقدر تفتحه بـ MeshLab أو CloudCompare أو Open3D.
    """
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_uint8):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"  PLY saved → {path}  ({len(points):,} points)")


# ─────────────────────────────────────────────
# Open3D — تفاعلي + فتح الـ PLY مباشرة
# ─────────────────────────────────────────────

def show_open3d(points, colors, out_dir, name):
    """
    بيبني الـ point cloud، بيحفظه كـ PLY،
    وبيفتحه مباشرة بنافذة Open3D تفاعلية.
    """
    try:
        import open3d as o3d

        print("\n  Building Open3D point cloud...")

        # بناء الـ point cloud
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

        # اعكس Y و Z عشان تطلع صح
        pcd.transform([[1, 0, 0, 0],
                       [0,-1, 0, 0],
                       [0, 0,-1, 0],
                       [0, 0, 0, 1]])

        # احفظ الـ PLY باستخدام Open3D مباشرة
        os.makedirs(out_dir, exist_ok=True)
        ply_path = os.path.join(out_dir, f"{name}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  PLY saved → {ply_path}  ({len(points):,} points)")

        # Controls
        print("\n  Open3D Controls:")
        print("    Mouse left drag  → rotate")
        print("    Mouse wheel      → zoom")
        print("    Mouse right drag → pan")
        print("    Press Q          → close\n")

        # ── افتح الـ PLY وعرضه مباشرة ────────────────────────────
        print("  Opening PLY in Open3D viewer...")
        loaded_pcd = o3d.io.read_point_cloud(ply_path)

        o3d.visualization.draw_geometries(
            [loaded_pcd],
            window_name = "3D Reconstruction — Monocular Depth Estimation",
            width       = 1280,
            height      = 720,
            point_show_normal = False,
        )

    except ImportError:
        print("\n  Open3D not installed.")
        print("  Install it: pip install open3d")
        print("  Then re-run to get the interactive viewer.\n")
        # Fallback: احفظ الـ PLY يدوياً
        ply_path = os.path.join(out_dir, f"{name}.ply")
        os.makedirs(out_dir, exist_ok=True)
        save_ply(points, colors, ply_path)
        print(f"  Open the .PLY file manually with MeshLab or CloudCompare.")


# ─────────────────────────────────────────────
# Matplotlib — 4 زوايا (للحفظ)
# ─────────────────────────────────────────────

def save_matplotlib(points, colors, rgb_np, depth_np,
                    max_points, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    # Subsample
    if len(points) > max_points:
        idx  = np.random.choice(len(points), max_points, replace=False)
        pts  = points[idx]
        cols = colors[idx]
    else:
        pts, cols = points, colors

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("3D Reconstruction from Monocular Depth Estimation",
                 fontsize=14, fontweight="bold")

    # RGB
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.clip(rgb_np, 0, 1))
    ax1.set_title("Input RGB Image", fontweight="bold")
    ax1.axis("off")

    # Depth map
    ax2 = fig.add_subplot(2, 3, 2)
    im  = ax2.imshow(depth_np, cmap="plasma")
    ax2.set_title("Predicted Depth Map", fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, label="Depth (m)")

    # 4 زوايا للـ point cloud
    views = [
        (2, 3, 3, 10,  0,  "Front View"),
        (2, 3, 4, 20, 60,  "Side View"),
        (2, 3, 5, 80,  0,  "Top View"),
        (2, 3, 6, 15, 30,  "Perspective View"),
    ]

    for row, col, pos, elev, azim, title in views:
        ax = fig.add_subplot(row, col, pos, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   c=cols, s=0.3, alpha=0.5)
        ax.set_title(f"3D Point Cloud — {title}", fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z / Depth (m)")
        ax.set_zlabel("Y (m)")
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{name}_3d.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matplotlib saved → {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  3D Reconstruction — Monocular Depth Estimation")
    print(f"{'='*55}")
    print(f"  Device : {device}\n")

    # ── Load model ────────────────────────────────────────────────
    model    = load_model(args.checkpoint, args.backbone, device)
    img_size = (args.img_h, args.img_w)

    # ── Get image ─────────────────────────────────────────────────
    if args.image:
        img_pil = Image.open(args.image).convert("RGB")
        name    = os.path.splitext(os.path.basename(args.image))[0]
    else:
        _, val_loader = get_dataloaders(
            data_root   = args.data,
            batch_size  = 1,
            img_size    = img_size,
            num_workers = 0,
        )
        dataset  = val_loader.dataset
        rgb_t, _ = dataset[args.idx]
        inv_mean = torch.tensor(NYU_MEAN).view(3, 1, 1)
        inv_std  = torch.tensor(NYU_STD ).view(3, 1, 1)
        rgb_disp = (rgb_t * inv_std + inv_mean).clamp(0, 1)
        img_pil  = T.ToPILImage()(rgb_disp)
        name     = f"sample_{args.idx:03d}"

    # ── Predict depth ─────────────────────────────────────────────
    tensor = preprocess(img_pil, img_size).to(device)
    with torch.no_grad():
        depth_pred = model(tensor)
    depth_np = depth_pred.squeeze().cpu().numpy()

    img_res  = img_pil.resize((args.img_w, args.img_h), Image.BILINEAR)
    rgb_np   = np.array(img_res, dtype=np.float32) / 255.0

    print(f"  Depth range: [{depth_np.min():.2f}, {depth_np.max():.2f}] m")

    # ── Build point cloud ─────────────────────────────────────────
    points, colors = depth_to_pointcloud(depth_np, rgb_np, args.img_h, args.img_w)
    print(f"  Points: {len(points):,}")

    # ── Save matplotlib figure ────────────────────────────────────
    print("\n  Saving matplotlib figure...")
    save_matplotlib(points, colors, rgb_np, depth_np,
                    args.max_points, args.out_dir, name)

    # ── Open3D: احفظ PLY وافتحه مباشرة ──────────────────────────
    if not args.no_open3d:
        show_open3d(points, colors, args.out_dir, name)

    print(f"\n  Done! Saved to: {args.out_dir}/\n")


if __name__ == "__main__":
    main()