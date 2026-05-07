"""
view_ply.py
-----------
عرض تفاعلي للـ PLY file بدون Open3D
باستخدام matplotlib فقط — تقدر تدوّره بالماوس

Usage:
    python view_ply.py results/3d/00008.ply
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_ply(path):
    points, colors = [], []
    with open(path) as f:
        # Skip header
        for line in f:
            if line.strip() == "end_header":
                break
        # Read points
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 6:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                colors.append([int(vals[3])/255, int(vals[4])/255, int(vals[5])/255])

    return np.array(points), np.array(colors)


def view(ply_path, max_pts=30000):
    print(f"  Loading: {ply_path}")
    points, colors = load_ply(ply_path)
    print(f"  Points: {len(points):,}")

    # Subsample
    if len(points) > max_pts:
        idx    = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#111111")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")

    ax.scatter(points[:, 0], points[:, 2], -points[:, 1],
               c=np.clip(colors, 0, 1), s=0.5, alpha=0.8)

    ax.set_title("3D Point Cloud — Drag to rotate", color="white",
                 fontsize=12, pad=15)
    ax.set_xlabel("X (m)", color="#888")
    ax.set_ylabel("Z / Depth (m)", color="#888")
    ax.set_zlabel("Y (m)", color="#888")
    ax.tick_params(colors="#555", labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#222")

    plt.tight_layout()
    print("  Drag with mouse to rotate. Close window to exit.")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/3d/00008.ply"
    view(path)