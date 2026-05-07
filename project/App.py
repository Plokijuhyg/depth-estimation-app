"""
App.py
------
DepthVision AI — Enhanced Streamlit Interface
==============================================
Features:
  • Animated particle / neural-grid background
  • Upload RGB image → depth estimation → interactive 3D reconstruction
  • Interactive Plotly 3D point cloud (rotate / zoom / pan in browser)
  • Depth overlay, error analysis, and download buttons
  • Integrates with model.py, visualize.py, Reconstruct_3d.py logic

Run:
    streamlit run App.py
"""

import os
import io
import time
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "DepthVision AI",
    page_icon   = "🔭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS + Animated Background ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #050508;
    --surface:  #0e0e16;
    --surface2: #13131f;
    --border:   #1a1a2e;
    --accent:   #7c6aff;
    --accent2:  #ff6a9e;
    --accent3:  #00d4ff;
    --gold:     #ffb347;
    --green:    #50c878;
    --text:     #e8e8f0;
    --muted:    #5a5a7a;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* ── Animated background canvas ── */
#depth-bg-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
    opacity: 0.55;
}

/* Push Streamlit content above canvas */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] { z-index: 2; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.8rem, 7vw, 5rem);
    letter-spacing: -3px;
    background: linear-gradient(135deg, #7c6aff 0%, #ff6a9e 45%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.05;
    margin-bottom: 0.4rem;
    animation: heroGlow 4s ease-in-out infinite alternate;
}
@keyframes heroGlow {
    from { filter: drop-shadow(0 0 20px rgba(124,106,255,0.3)); }
    to   { filter: drop-shadow(0 0 40px rgba(0,212,255,0.4)); }
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-block;
    background: rgba(124,106,255,0.12);
    border: 1px solid rgba(124,106,255,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 1px;
    margin: 3px;
    transition: all 0.2s;
}
.badge:hover {
    background: rgba(124,106,255,0.25);
    border-color: rgba(124,106,255,0.5);
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,106,255,0.4), transparent);
}
.card:hover {
    border-color: rgba(124,106,255,0.35);
    box-shadow: 0 0 30px rgba(124,106,255,0.06);
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 1rem;
}
@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
.metric-box {
    background: linear-gradient(135deg, rgba(124,106,255,0.06), rgba(0,212,255,0.04));
    border: 1px solid rgba(124,106,255,0.18);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    animation: fadeInUp 0.4s ease both;
}
.metric-box:hover { transform: translateY(-2px); border-color: rgba(124,106,255,0.4); }
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent3);
    line-height: 1;
}
.metric-lbl {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

/* ── Pulse loader ── */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.6); }
}

/* ── Upload area ── */
[data-testid="stFileUploader"] > div {
    background: rgba(124,106,255,0.04) !important;
    border: 2px dashed rgba(124,106,255,0.25) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(124,106,255,0.5) !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c6aff, #ff6a9e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,106,255,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text) !important; }

/* ── Status / info bars ── */
.status-bar {
    background: rgba(0,212,255,0.07);
    border-left: 3px solid var(--accent3);
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent3);
    margin: 1rem 0;
    animation: slideIn 0.3s ease;
}
@keyframes slideIn {
    from { opacity:0; transform:translateX(-10px); }
    to   { opacity:1; transform:translateX(0); }
}
.info-box {
    background: rgba(255,179,71,0.07);
    border: 1px solid rgba(255,179,71,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--gold);
    margin: 0.5rem 0;
}
.success-box {
    background: rgba(80,200,120,0.07);
    border: 1px solid rgba(80,200,120,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--green);
    margin: 0.5rem 0;
}
.error-box {
    background: rgba(255,106,158,0.07);
    border: 1px solid rgba(255,106,158,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--accent2);
    margin: 0.5rem 0;
}

/* ── Image labels ── */
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 6px;
}
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Result section header ── */
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0 1rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: var(--text);
}
.result-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--accent), transparent);
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
[data-testid="stRadio"] label { color: var(--text) !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    letter-spacing: 1px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
</style>

<!-- Animated Background Canvas -->
<canvas id="depth-bg-canvas"></canvas>
<script>
(function() {
    const canvas = document.getElementById('depth-bg-canvas');
    const ctx    = canvas.getContext('2d');
    let W, H, particles, grid;

    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    // Particles
    const N = 60;
    particles = Array.from({length: N}, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.5 + 0.5,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        hue: Math.random() > 0.6 ? 260 : (Math.random() > 0.5 ? 200 : 330),
    }));

    // Grid lines
    const GRID_COLS = 20;
    const GRID_ROWS = 14;

    let frame = 0;
    function draw() {
        ctx.clearRect(0, 0, W, H);
        frame++;

        // ── Neural grid ──
        const cw = W / GRID_COLS;
        const ch = H / GRID_ROWS;

        ctx.lineWidth = 0.4;
        for (let r = 0; r <= GRID_ROWS; r++) {
            const y   = r * ch;
            const osc = Math.sin(frame * 0.012 + r * 0.4) * 4;
            ctx.beginPath();
            ctx.moveTo(0, y + osc);
            for (let c = 1; c <= GRID_COLS; c++) {
                const x  = c * cw;
                const dy = Math.sin(frame * 0.010 + c * 0.3 + r * 0.5) * 6;
                ctx.lineTo(x, y + osc + dy);
            }
            const alpha = 0.08 + 0.04 * Math.sin(frame * 0.008 + r);
            ctx.strokeStyle = `rgba(124,106,255,${alpha})`;
            ctx.stroke();
        }
        for (let c = 0; c <= GRID_COLS; c++) {
            const x   = c * cw;
            const osc = Math.sin(frame * 0.010 + c * 0.35) * 4;
            ctx.beginPath();
            ctx.moveTo(x + osc, 0);
            for (let r = 1; r <= GRID_ROWS; r++) {
                const y  = r * ch;
                const dx = Math.sin(frame * 0.012 + r * 0.25 + c * 0.4) * 5;
                ctx.lineTo(x + osc + dx, y);
            }
            const alpha = 0.05 + 0.03 * Math.sin(frame * 0.009 + c);
            ctx.strokeStyle = `rgba(0,212,255,${alpha})`;
            ctx.stroke();
        }

        // ── Scanning line ──
        const scanY = (H * 0.5) + Math.sin(frame * 0.015) * (H * 0.4);
        const scanGrad = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 40);
        scanGrad.addColorStop(0,   'rgba(124,106,255,0)');
        scanGrad.addColorStop(0.5, 'rgba(124,106,255,0.06)');
        scanGrad.addColorStop(1,   'rgba(124,106,255,0)');
        ctx.fillStyle = scanGrad;
        ctx.fillRect(0, scanY - 40, W, 80);

        // ── Particles ──
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = W;
            if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H;
            if (p.y > H) p.y = 0;

            const pulse = 0.5 + 0.5 * Math.sin(frame * 0.04 + p.x);
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue},80%,70%,${0.35 * pulse})`;
            ctx.fill();

            // Connection lines
            particles.forEach(q => {
                const dist = Math.hypot(p.x - q.x, p.y - q.y);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(q.x, q.y);
                    ctx.strokeStyle = `rgba(124,106,255,${0.08 * (1 - dist/100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NYU_MEAN = [0.485, 0.456, 0.406]
NYU_STD  = [0.229, 0.224, 0.225]
MAX_DEPTH = 10.0
NYU_FX, NYU_FY = 518.8579, 519.4696
NYU_CX, NYU_CY = 325.5824, 253.7362
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str, backbone: str = "resnet34"):
    try:
        from model import DepthEstimationNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DepthEstimationNet(backbone=backbone, pretrained=False).to(device)
        ckpt   = torch.load(checkpoint_path, map_location=device)
        state  = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        return model, device, None, epoch
    except FileNotFoundError:
        return None, None, f"Checkpoint not found: {checkpoint_path}", None
    except Exception as e:
        return None, None, str(e), None

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img_pil: Image.Image, img_h: int, img_w: int):
    img_r  = img_pil.resize((img_w, img_h), Image.BILINEAR)
    tensor = TF.to_tensor(img_r)
    tensor = T.Normalize(mean=NYU_MEAN, std=NYU_STD)(tensor)
    return tensor.unsqueeze(0), img_r


def predict_depth(model, device, img_pil, img_h, img_w):
    tensor, img_r = preprocess(img_pil, img_h, img_w)
    tensor = tensor.to(device)
    with torch.no_grad():
        depth = model(tensor)
    depth_np = depth.squeeze().cpu().numpy()
    rgb_np   = np.array(img_r, dtype=np.float32) / 255.0
    return depth_np, rgb_np


def depth_colormap(depth_np: np.ndarray, cmap: str = "jet") -> np.ndarray:
    vmin = depth_np.min(); vmax = depth_np.max()
    norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap_fn = getattr(cm, cmap)
    return (cmap_fn(norm)[:, :, :3] * 255).astype(np.uint8)


def depth_overlay(rgb_np: np.ndarray, depth_np: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    norm  = np.clip((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8), 0, 1)
    d_col = cm.jet(norm)[:, :, :3]
    blend = alpha * d_col + (1 - alpha) * np.clip(rgb_np, 0, 1)
    return np.clip(blend * 255, 0, 255).astype(np.uint8)


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#050508")
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────────────────────────────────────
# Point cloud
# ─────────────────────────────────────────────────────────────────────────────
def depth_to_pointcloud(depth_np, rgb_np, img_h, img_w):
    fx = NYU_FX * img_w / 640.0
    fy = NYU_FY * img_h / 480.0
    cx = NYU_CX * img_w / 640.0
    cy = NYU_CY * img_h / 480.0

    u = np.arange(img_w, dtype=np.float32)
    v = np.arange(img_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z      = depth_np
    X      = (uu - cx) * Z / fx
    Y      = (vv - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb_np.reshape(-1, 3)
    mask   = (Z.flatten() > 0.1) & (Z.flatten() < MAX_DEPTH)
    return points[mask], colors[mask]


def make_plotly_3d(points, colors, max_pts: int = 40000):
    """Return an interactive Plotly 3D scatter figure."""
    import plotly.graph_objects as go

    if len(points) > max_pts:
        idx    = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Flip Y for right-hand convention
    X =  points[:, 0]
    Y = -points[:, 1]
    Z = -points[:, 2]   # negate depth so near=front, far=back

    cols_rgb = np.clip(colors, 0, 1)
    col_strs = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                for r, g, b in cols_rgb]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=X, y=Z, z=Y,          # z=depth axis in Plotly layout
            mode='markers',
            marker=dict(
                size        = 1.2,
                color       = col_strs,
                opacity     = 0.85,
            ),
            hovertemplate = (
                "X: %{x:.2f}m<br>"
                "Depth: %{y:.2f}m<br>"
                "Y: %{z:.2f}m<extra></extra>"
            ),
        )]
    )
    fig.update_layout(
        paper_bgcolor = '#050508',
        plot_bgcolor  = '#050508',
        scene = dict(
            bgcolor    = '#0e0e16',
            xaxis = dict(
                title       = 'X (m)',
                color       = '#5a5a7a',
                gridcolor   = '#1a1a2e',
                zerolinecolor='#1a1a2e',
                showbackground=True,
                backgroundcolor='#0e0e16',
            ),
            yaxis = dict(
                title       = 'Depth / Z (m)',
                color       = '#5a5a7a',
                gridcolor   = '#1a1a2e',
                zerolinecolor='#1a1a2e',
                showbackground=True,
                backgroundcolor='#0e0e16',
            ),
            zaxis = dict(
                title       = 'Y (m)',
                color       = '#5a5a7a',
                gridcolor   = '#1a1a2e',
                zerolinecolor='#1a1a2e',
                showbackground=True,
                backgroundcolor='#0e0e16',
            ),
            camera = dict(
                eye=dict(x=1.4, y=-1.2, z=0.8),
            ),
        ),
        margin = dict(l=0, r=0, t=0, b=0),
        height = 600,
        font   = dict(family='Space Mono, monospace', color='#5a5a7a', size=11),
    )
    return fig


def make_depth_figure(rgb_np, depth_np):
    """Matplotlib figure for download."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#050508")

    for ax in axes:
        ax.set_facecolor("#0e0e16")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a2e")

    axes[0].imshow(np.clip(rgb_np, 0, 1))
    axes[0].set_title("Input RGB", color="#e8e8f0", fontsize=11, fontfamily="monospace", pad=10)
    axes[0].axis("off")

    im = axes[1].imshow(depth_np, cmap="jet", vmin=depth_np.min(), vmax=depth_np.max())
    axes[1].set_title("Predicted Depth Map", color="#e8e8f0", fontsize=11, fontfamily="monospace", pad=10)
    axes[1].axis("off")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("Depth (m)", color="#5a5a7a", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#5a5a7a")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#5a5a7a")

    # Overlay
    norm  = np.clip((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8), 0, 1)
    d_col = cm.jet(norm)[:, :, :3]
    blend = np.clip(0.55 * d_col + 0.45 * rgb_np, 0, 1)
    axes[2].imshow(blend)
    axes[2].set_title("Depth Overlay", color="#e8e8f0", fontsize=11, fontfamily="monospace", pad=10)
    axes[2].axis("off")

    plt.tight_layout(pad=1.5)
    return fig


def save_ply(points, colors) -> bytes:
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    buf = io.StringIO()
    buf.write("ply\nformat ascii 1.0\n")
    buf.write(f"element vertex {len(points)}\n")
    buf.write("property float x\nproperty float y\nproperty float z\n")
    buf.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    buf.write("end_header\n")
    for p, c in zip(points, colors_u8):
        buf.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    return buf.getvalue().encode()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                    color:#5a5a7a;letter-spacing:4px;text-transform:uppercase;
                    margin-bottom:0.8rem'>Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">⚙️ Model</div>', unsafe_allow_html=True)

    backbone   = st.selectbox("Backbone", ["resnet34", "resnet18"], index=0)
    ckpt_path  = st.text_input("Checkpoint", value=CHECKPOINT_PATH)
    img_h = st.select_slider("Height (px)", options=[240, 480], value=480)
    img_w = st.select_slider("Width  (px)", options=[320, 640], value=640)

    st.markdown("<hr style='border-color:#1a1a2e;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎨 Visualisation</div>', unsafe_allow_html=True)

    depth_cmap   = st.selectbox("Depth colormap", ["jet", "turbo", "plasma", "inferno", "magma", "viridis"], index=0)
    overlay_alpha= st.slider("Overlay alpha", 0.3, 0.8, 0.55, step=0.05)
    max_pts_3d   = st.select_slider("Max 3D points", options=[10000, 20000, 40000, 60000, 80000], value=40000)

    st.markdown("<hr style='border-color:#1a1a2e;margin:1rem 0'>", unsafe_allow_html=True)

    # Device + version info
    dev_name  = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    dev_color = "#50c878" if torch.cuda.is_available() else "#ffb347"
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#5a5a7a;line-height:2'>
        Device: <span style='color:{dev_color}'>{dev_name}</span><br>
        PyTorch: <span style='color:#7c6aff'>{torch.__version__}</span><br>
        Arch: <span style='color:#00d4ff'>ResNet-34 + U-Net</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a1a2e;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#5a5a7a;line-height:1.9'>
        <div style='color:#7c6aff;margin-bottom:4px;font-size:0.72rem'>Architecture</div>
        Encoder: ResNet-34<br>
        Decoder: U-Net + Skips<br>
        Params: ~24.5M<br>
        Dataset: NYU Depth V2<br>
        Loss: L1 + SILog + Gradient<br>
        Output: Dense depth map
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">DepthVision AI</div>
    <div class="hero-sub">Monocular Depth Estimation · 3D Reconstruction</div>
    <div>
        <span class="badge">ResNet-34</span>
        <span class="badge">U-Net Decoder</span>
        <span class="badge">NYU Depth V2</span>
        <span class="badge">SILog Loss</span>
        <span class="badge">PyTorch</span>
        <span class="badge">Plotly 3D</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
model, device, model_error, ckpt_epoch = load_model(ckpt_path, backbone)

if model_error:
    st.markdown(f"""
    <div class="error-box">
        ⚠️ <strong>Model not loaded:</strong> {model_error}<br>
        Make sure <code>{ckpt_path}</code> exists.<br>
        Run training first: <code>python train.py --data dataset/ --epochs 20</code>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-box">
        ✓ Model loaded from epoch {ckpt_epoch} &nbsp;·&nbsp;
        <code>{ckpt_path}</code> &nbsp;·&nbsp; {dev_name}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Upload + Mode
# ─────────────────────────────────────────────────────────────────────────────
col_up, col_mode = st.columns([1.3, 1])

with col_up:
    st.markdown('<div class="card-title">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop an indoor RGB image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

with col_mode:
    st.markdown('<div class="card-title">🎯 Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        options=[
            "🔍 Depth Estimation Only",
            "🌐 3D Reconstruction Only",
            "✨ Both (Depth + 3D)",
        ],
        index=2,
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶  Run Analysis",
        disabled=(model is None or uploaded is None),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Image preview
# ─────────────────────────────────────────────────────────────────────────────
if uploaded and not run_btn:
    img_prev = Image.open(uploaded).convert("RGB")
    st.markdown('<div class="card-title" style="margin-top:1rem">🖼️ Input Preview</div>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns([1, 2, 1])
    with p2:
        st.image(img_prev, use_container_width=True)
        st.markdown(
            f'<div class="img-label">{uploaded.name} — {img_prev.width}×{img_prev.height}px</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and uploaded and model:
    uploaded.seek(0)
    img_pil  = Image.open(uploaded).convert("RGB")
    do_depth = "Depth" in mode or "Both" in mode
    do_3d    = "3D"    in mode or "Both" in mode

    img_h = 480
    img_w = 640

    # ── Step 1: Depth inference ──────────────────────────────────────
    with st.spinner(""):
        st.markdown(
            '<div class="status-bar"><span class="pulse-dot"></span>'
            'Running depth estimation neural network…</div>',
            unsafe_allow_html=True,
        )
        t0 = time.time()
        depth_np, rgb_np = predict_depth(model, device, img_pil, img_h, img_w)
        elapsed = time.time() - t0

    st.markdown(f"""
    <div class="status-bar">
        ✓ Inference complete &nbsp;·&nbsp; {elapsed:.2f}s &nbsp;·&nbsp;
        Depth range: {depth_np.min():.2f}m – {depth_np.max():.2f}m &nbsp;·&nbsp;
        Resolution: {img_w}×{img_h}
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────────
    d_mean = depth_np.mean()
    d_std  = depth_np.std()
    d_min  = depth_np.min()
    d_max  = depth_np.max()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val">{d_mean:.2f}m</div>
            <div class="metric-lbl">Mean Depth</div>
        </div>
        <div class="metric-box" style="animation-delay:0.05s">
            <div class="metric-val">{d_max - d_min:.2f}m</div>
            <div class="metric-lbl">Depth Range</div>
        </div>
        <div class="metric-box" style="animation-delay:0.1s">
            <div class="metric-val">{d_min:.2f}m</div>
            <div class="metric-lbl">Min Depth</div>
        </div>
        <div class="metric-box" style="animation-delay:0.15s">
            <div class="metric-val">{d_std:.2f}m</div>
            <div class="metric-lbl">Depth Std Dev</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Depth section ────────────────────────────────────────────────
    if do_depth:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-header">
            🔍 Depth Estimation
            <div class="result-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        tab_vis, tab_cmap, tab_download = st.tabs(
            ["📸 Visualisation", "🎨 Colormaps", "⬇ Download"]
        )

        with tab_vis:
            c1, c2, c3 = st.columns(3)
            img_resized = img_pil.resize((img_w, img_h), Image.BILINEAR)
            with c1:
                st.image(img_resized, use_container_width=True)
                st.markdown('<div class="img-label">Input RGB</div>', unsafe_allow_html=True)
            with c2:
                d_col = depth_colormap(depth_np, depth_cmap)
                st.image(d_col, use_container_width=True)
                st.markdown(f'<div class="img-label">Depth Map ({depth_cmap})</div>', unsafe_allow_html=True)
            with c3:
                ov = depth_overlay(rgb_np, depth_np, alpha=overlay_alpha)
                st.image(ov, use_container_width=True)
                st.markdown('<div class="img-label">Depth Overlay</div>', unsafe_allow_html=True)

        with tab_cmap:
            st.markdown(
                '<p style="font-family:Space Mono,monospace;font-size:0.78rem;color:#5a5a7a;">'
                'Depth visualised with 4 different colormaps</p>',
                unsafe_allow_html=True,
            )
            row = st.columns(4)
            for ax_i, cname in enumerate(["jet", "turbo", "plasma", "inferno"]):
                with row[ax_i]:
                    cm_img = depth_colormap(depth_np, cname)
                    st.image(cm_img, use_container_width=True)
                    st.markdown(f'<div class="img-label">{cname}</div>', unsafe_allow_html=True)

        with tab_download:
            fig_dl  = make_depth_figure(rgb_np, depth_np)
            pil_dl  = fig_to_pil(fig_dl)
            plt.close(fig_dl)
            buf_dl  = io.BytesIO()
            pil_dl.save(buf_dl, format="PNG")
            st.download_button(
                label     = "⬇  Download Depth Figure (PNG)",
                data      = buf_dl.getvalue(),
                file_name = "depth_result.png",
                mime      = "image/png",
            )
            # Raw depth as .npy
            npy_buf = io.BytesIO()
            np.save(npy_buf, depth_np)
            st.download_button(
                label     = "⬇  Download Raw Depth Array (.npy)",
                data      = npy_buf.getvalue(),
                file_name = "depth_map.npy",
                mime      = "application/octet-stream",
            )

    # ── 3D Reconstruction section ────────────────────────────────────
    if do_3d:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-header">
            🌐 Interactive 3D Reconstruction
            <div class="result-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(""):
            st.markdown(
                '<div class="status-bar"><span class="pulse-dot"></span>'
                'Building point cloud from depth map…</div>',
                unsafe_allow_html=True,
            )
            t1 = time.time()
            points, colors = depth_to_pointcloud(depth_np, rgb_np, img_h, img_w)
            t2 = time.time()

        st.markdown(f"""
        <div class="status-bar">
            ✓ Point cloud built &nbsp;·&nbsp; {len(points):,} 3D points &nbsp;·&nbsp;
            {t2-t1:.2f}s &nbsp;·&nbsp;
            Rendered: {min(len(points), max_pts_3d):,} pts
        </div>
        """, unsafe_allow_html=True)

        # Interactive Plotly viewer
        st.markdown("""
        <div class="info-box" style="margin-bottom:1rem">
            🖱️ <strong>Interact:</strong>
            Left-drag to rotate &nbsp;·&nbsp;
            Scroll to zoom &nbsp;·&nbsp;
            Right-drag to pan &nbsp;·&nbsp;
            Double-click to reset
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Rendering interactive 3D viewer…"):
            fig_3d = make_plotly_3d(points, colors, max_pts=max_pts_3d)

        st.plotly_chart(fig_3d, use_container_width=True)

        # Download row
        dl1, dl2 = st.columns(2)
        with dl1:
            ply_data = save_ply(points, colors)
            st.download_button(
                label     = "⬇  Download .PLY Point Cloud",
                data      = ply_data,
                file_name = "pointcloud.ply",
                mime      = "application/octet-stream",
            )
        with dl2:
            # Static matplotlib 4-view figure
            with st.spinner("Generating static 4-view figure…"):
                fig_static = plt.figure(figsize=(20, 8))
                fig_static.patch.set_facecolor("#050508")
                subpts = min(len(points), 20000)
                idx_s  = np.random.choice(len(points), subpts, replace=False)
                pts_s  = points[idx_s]; col_s = colors[idx_s]
                views4 = [(10, 0, "Front"), (20, 60, "Side"), (70, 30, "Top"), (15, 120, "Perspective")]
                for vi, (elev, azim, vtitle) in enumerate(views4, 1):
                    ax3 = fig_static.add_subplot(1, 4, vi, projection="3d")
                    ax3.set_facecolor("#0e0e16")
                    ax3.scatter(pts_s[:,0], -pts_s[:,2], -pts_s[:,1],
                                c=np.clip(col_s,0,1), s=0.3, alpha=0.6)
                    ax3.set_title(vtitle, color="#e8e8f0", fontsize=9,
                                  fontfamily="monospace", pad=6)
                    ax3.set_xlabel("X", color="#5a5a7a", fontsize=7)
                    ax3.set_ylabel("Z", color="#5a5a7a", fontsize=7)
                    ax3.set_zlabel("Y", color="#5a5a7a", fontsize=7)
                    ax3.tick_params(colors="#5a5a7a", labelsize=6)
                    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
                        pane.fill = False; pane.set_edgecolor("#1a1a2e")
                    ax3.view_init(elev=elev, azim=azim)
                plt.tight_layout(pad=1)
                pil_4v = fig_to_pil(fig_static)
                plt.close(fig_static)

            buf_4v = io.BytesIO()
            pil_4v.save(buf_4v, format="PNG")
            st.download_button(
                label     = "⬇  Download 4-View Static Figure",
                data      = buf_4v.getvalue(),
                file_name = "reconstruction_3d.png",
                mime      = "image/png",
            )

        st.markdown("""
        <div class="info-box">
            💡 Open the <strong>.PLY file</strong> in
            <strong>MeshLab</strong> or <strong>CloudCompare</strong>
            for a full desktop interactive viewer with unlimited zoom.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:3rem 0 1rem;
            font-family:Space Mono,monospace;font-size:0.65rem;color:#2a2a4a'>
    DepthVision AI &nbsp;·&nbsp; Monocular Depth Estimation &nbsp;·&nbsp;
    ResNet-34 + U-Net &nbsp;·&nbsp; NYU Depth V2 &nbsp;·&nbsp;
    SILog + L1 + Gradient Loss
</div>
""", unsafe_allow_html=True)