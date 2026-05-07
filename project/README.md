# Monocular Depth Estimation using Deep Learning (3D Reconstruction)

> **University Project** | Computer Vision | Deep Learning  
> Dataset: NYU Depth V2 | Framework: PyTorch | Architecture: ResNet-34 + U-Net Decoder

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Model Architecture](#4-model-architecture)
5. [Loss Functions](#5-loss-functions)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Environment Setup](#7-environment-setup)
8. [Training](#8-training)
9. [Testing & Inference](#9-testing--inference)
10. [Visualisation](#10-visualisation)
11. [Expected Results](#11-expected-results)
12. [Scale Ambiguity — Theoretical Background](#12-scale-ambiguity--theoretical-background)
13. [Encoder–Decoder Advantages](#13-encoderdecoder-advantages)
14. [Future Improvements](#14-future-improvements)
15. [References](#15-references)

---

## 1. Project Overview

**Objective:** Given a single RGB photograph, predict a dense, pixel-wise depth map that encodes the 3D structure of the scene.

### Why is this hard?

A standard camera performs a *lossy* projection: 3D world coordinates are collapsed onto a 2D image plane. Recovering the third dimension from a single image requires the model to learn rich scene priors — object sizes, perspective cues, texture gradients, and shading — all of which are implicit in images but never directly encoded.

### What we build

| Component | Choice |
|-----------|--------|
| Encoder | ResNet-34 (pretrained on ImageNet) |
| Decoder | U-Net style with 5 upsampling stages |
| Skip connections | Yes — from every encoder stage |
| Loss | L1 + Scale-Invariant Log + Gradient |
| Dataset | NYU Depth V2 (1,449 labelled indoor RGBD pairs) |
| Output | Single-channel depth map (metres) |

---

## 2. Folder Structure

```
project/
│
├── dataset/                     ← Place your HDF5/MAT file here
│   └── nyu_depth_v2_labeled.mat
│
├── checkpoints/                 ← Saved model weights (auto-created)
│   ├── best_model.pth
│   └── train_log.csv
│
├── results/                     ← Visualisation outputs (auto-created)
│
├── models/                      ← (optional) extra architecture variants
│
├── dataset_loader.py            ← NYU HDF5 parser + PyTorch Dataset
├── model.py                     ← ResNet-34 encoder + U-Net decoder
├── loss.py                      ← L1, SILog, Gradient losses
├── utils.py                     ← Metrics, checkpoint helpers
├── train.py                     ← Full training loop
├── test.py                      ← Evaluation + single-image inference
├── visualize.py                 ← All plotting utilities
├── requirements.txt
└── README.md
```

---

## 3. Dataset Preparation

### Step 1 — Install Kaggle CLI

```bash
pip install kaggle
```

Configure your API key:

1. Log in to [kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New API Token**
3. This downloads `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Step 2 — Download NYU Depth V2

```bash
mkdir -p dataset
kaggle datasets download -d soumikrakshit/nyu-depth-v2 -p dataset/
```

### Step 3 — Extract

```bash
cd dataset
unzip nyu-depth-v2.zip
cd ..
```

You should now have:
```
dataset/
└── nyu_depth_v2_labeled.mat   (~2.8 GB)
```

> **Alternative (manual):** Download from [Kaggle dataset page](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2), extract, and place in `dataset/`.

### Step 4 — Verify the file

```python
import h5py
with h5py.File("dataset/nyu_depth_v2_labeled.mat", "r") as f:
    print(list(f.keys()))   # should show: images, depths (and others)
    print(f["images"].shape)  # (1449, 3, 480, 640)
    print(f["depths"].shape)  # (1449, 480, 640)
```

### Preprocessing Pipeline

The `dataset_loader.py` performs the following steps automatically:

| Step | Description |
|------|-------------|
| Load HDF5 | Read images `(N,3,H,W)` and depths `(N,H,W)` |
| Train/val split | 80% / 20% deterministic split |
| Resize | Default 240×320 (configurable) |
| Normalise images | ImageNet mean/std subtraction |
| Clip depth | Valid range: [0.001, 10.0] metres |
| Augmentation (train) | H-flip, colour jitter, random crop |

---

## 4. Model Architecture

```
INPUT: RGB image (3 × 240 × 320)
│
├─── ENCODER (ResNet-34, pretrained)
│    │
│    ├── Stage 0: Conv7×7 + BN + ReLU      →  (64,  120, 160)  ──────────────────┐ skip_0
│    ├── MaxPool                             →  (64,   60,  80)                   │
│    ├── Stage 1: ResBlocks × 3             →  (64,   60,  80)  ────────────────┐ │ skip_1
│    ├── Stage 2: ResBlocks × 4 (stride 2)  →  (128,  30,  40)  ──────────────┐ │ │ skip_2
│    ├── Stage 3: ResBlocks × 6 (stride 2)  →  (256,  15,  20)  ────────────┐ │ │ │ skip_3
│    └── Stage 4: ResBlocks × 3 (stride 2)  →  (512,   8,  10)  (bottleneck) │ │ │ │
│                                                                              │ │ │ │
└─── DECODER (U-Net style)                                                    │ │ │ │
     │                                                                         │ │ │ │
     ├── UpBlock 1: bilinear×2 + cat(skip_3) →  (256,  15,  20)  ←───────────┘ │ │ │
     ├── UpBlock 2: bilinear×2 + cat(skip_2) →  (128,  30,  40)  ←─────────────┘ │ │
     ├── UpBlock 3: bilinear×2 + cat(skip_1) →  (64,   60,  80)  ←───────────────┘ │
     ├── UpBlock 4: bilinear×2 + cat(skip_0) →  (64,  120, 160)  ←─────────────────┘
     └── UpBlock 5: bilinear×2               →  (32,  240, 320)
          │
          └── HEAD: Conv1×1 → Sigmoid × 10.0 →  (1,   240, 320)  DEPTH MAP
```

### Why skip connections?

When the encoder compresses the image down to a 8×10 bottleneck, fine spatial details (edges, corners, object boundaries) are lost. Skip connections bring these fine-grained features back into the decoder at every resolution level, resulting in sharper and geometrically more accurate depth maps.

Without skip connections the decoder produces over-smoothed, blurry depth — a well-known failure mode of plain encoder–decoder architectures.

---

## 5. Loss Functions

Three losses are combined:

### 5.1 L1 Loss (Masked)

Measures absolute pixel-wise error in metres, ignoring invalid depth pixels (sensor dropout):

```
L1 = (1/n) Σ |pred_i − gt_i|
```

Penalises large errors proportionally; more robust to outliers than L2.

### 5.2 Scale-Invariant Logarithmic Loss (SILog)

Addresses scale ambiguity by operating in log-depth space:

```
d_i  = log(pred_i) − log(gt_i)
SILog = (1/n) Σ d_i²  −  (λ/n²) (Σ d_i)²
```

With λ = 0.5 this subtracts the mean log-difference, making the loss **invariant to a global scale factor**. A prediction that is uniformly 2× too large is penalised much less than one with incorrect *relative* structure.

### 5.3 Gradient Loss

Encourages spatial sharpness at depth boundaries:

```
L_grad = mean(|∇x pred − ∇x gt| + |∇y pred − ∇y gt|)
```

### Combined

```
L_total = α * L1  +  β * SILog  +  γ * L_grad
          1.0         1.0           0.5        (defaults)
```

---

## 6. Evaluation Metrics

All metrics are computed only over **valid pixels** (gt > 0.001 m).

| Metric | Formula | Better |
|--------|---------|--------|
| **RMSE** | √(mean((pred−gt)²)) | ↓ lower |
| **MAE** | mean(|pred−gt|) | ↓ lower |
| **AbsRel** | mean(|pred−gt| / gt) | ↓ lower |
| **SqRel** | mean((pred−gt)² / gt) | ↓ lower |
| **RMSE_log** | √(mean((log pred − log gt)²)) | ↓ lower |
| **δ < 1.25** | % pixels: max(p/g, g/p) < 1.25 | ↑ higher |
| **δ < 1.25²** | same, threshold 1.5625 | ↑ higher |
| **δ < 1.25³** | same, threshold 1.9531 | ↑ higher |

### Typical results on NYU Depth V2 (after ~30 epochs):

| RMSE | AbsRel | δ < 1.25 |
|------|--------|---------|
| ~0.55 m | ~0.14 | ~78% |

---

## 7. Environment Setup

### Requirements

- Python 3.9+
- CUDA 11.7+ (optional but strongly recommended)

### Install

```bash
# Clone / download project
cd project/

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify CUDA

```python
import torch
print(torch.cuda.is_available())   # True if GPU is available
print(torch.cuda.get_device_name(0))
```

---

## 8. Training

### Quick start

```bash
python train.py --data dataset/nyu_depth_v2_labeled.mat
```

### Full options

```bash
python train.py \
    --data       dataset/nyu_depth_v2_labeled.mat \
    --backbone   resnet34 \
    --epochs     30 \
    --batch-size 8 \
    --lr         1e-4 \
    --img-h      240 \
    --img-w      320 \
    --out-dir    checkpoints \
    --workers    4
```

### Resume from checkpoint

```bash
python train.py \
    --data     dataset/nyu_depth_v2_labeled.mat \
    --resume   checkpoints/checkpoint_epoch015.pth
```

### Training on CPU (slow, for testing)

```bash
python train.py --data dataset/nyu_depth_v2_labeled.mat \
                --epochs 2 --batch-size 2 --workers 0
```

### Training outputs

```
checkpoints/
├── best_model.pth              ← Best validation RMSE checkpoint
├── checkpoint_epoch030.pth     ← Latest epoch checkpoint
└── train_log.csv               ← Per-epoch metrics CSV
```

---

## 9. Testing & Inference

### Evaluate on the validation set

```bash
python test.py \
    --checkpoint checkpoints/best_model.pth \
    --data       dataset/nyu_depth_v2_labeled.mat \
    --out-dir    results/
```

Output:
```
──────────────────────────────────────────────────
  Test / Validation Metrics
──────────────────────────────────────────────────
  Error metrics (lower ↓ is better):
    RMSE         : 0.5512
    MAE          : 0.3891
    AbsRel       : 0.1402
    SqRel        : 0.0821
    RMSE_log     : 0.1934
  Threshold accuracy (higher ↑ is better):
    δ < 1.25     : 78.34 %
    δ < 1.25²    : 94.11 %
    δ < 1.25³    : 98.20 %
──────────────────────────────────────────────────
```

### Single-image inference

```bash
python test.py \
    --checkpoint checkpoints/best_model.pth \
    --image      path/to/your/image.jpg \
    --out-dir    results/
```

Produces `results/<imagename>_depth.png`.

---

## 10. Visualisation

Generate all visualisations from a trained model:

```bash
python visualize.py \
    --checkpoint checkpoints/best_model.pth \
    --data       dataset/nyu_depth_v2_labeled.mat \
    --log        checkpoints/train_log.csv \
    --out-dir    results/visualisations \
    --n-samples  8
```

Outputs per sample:
- `sample_001_comparison.png` — RGB | GT depth | Predicted depth
- `sample_001_error.png`      — Absolute and relative error maps
- `sample_001_overlay.png`    — Depth colormap overlaid on RGB
- `depth_histogram.png`       — Distribution of GT vs predicted depths
- `training_curves.png`       — Loss / RMSE / δ curves over epochs

---

## 11. Expected Results

### After 30 epochs on NYU Depth V2 (ResNet-34 encoder, 240×320)

| Metric | Expected Value |
|--------|---------------|
| RMSE | ~0.50–0.60 m |
| MAE | ~0.35–0.45 m |
| AbsRel | ~0.13–0.16 |
| δ < 1.25 | ~75–82% |
| Training time (GPU) | ~3–6 hours |
| Training time (CPU) | ~40–80 hours |

---

## 12. Scale Ambiguity — Theoretical Background

### What is scale ambiguity?

Consider two photos taken at different distances from a room:
- A photo of a **1:10 scale model** of a bedroom
- A photo of a **real bedroom**

If the camera is positioned proportionally, the pixel-level appearance can be **identical**. The images encode only *angular relationships* — nothing about absolute distance.

This is the **monocular depth scale ambiguity problem**: a single 2D image provides no direct metric scale information.

### Mathematical statement

If depth map `D` is a valid explanation for image `I`, then so is `α·D` for any scalar `α > 0`. The image formation model is:

```
I(x,y) = f(scene(x·D/f, y·D/f, D))
```

Scaling `D` by `α` and the camera focal length by the same factor produces an identical image.

### How we address it

1. **Large, consistent dataset**: NYU Depth V2 contains ~1,449 calibrated indoor scenes with metric ground-truth from a Kinect depth sensor. The network learns a strong **scale prior** from the data distribution (indoor rooms tend to be 2–10m deep).

2. **Scale-Invariant Loss (SILog)**: By computing loss in log-depth space and subtracting the mean log-difference, SILog tolerates a global scale offset between prediction and ground truth, focusing learning on *relative* depth structure.

3. **Absolute scale from context**: The pretrained ResNet encoder brings knowledge of object sizes (chairs, beds, doors) from ImageNet, providing implicit absolute scale cues.

### Why this matters

In **stereo vision** or **LiDAR-assisted** depth estimation, absolute scale is trivially available from the baseline or sensor calibration. Monocular depth estimation must infer it — making it fundamentally harder and more interesting as a research problem.

---

## 13. Encoder–Decoder Advantages

### Why not a plain CNN regressor?

A plain CNN (no decoder) would output a feature vector and then upsample once. This approach loses spatial correspondence — the network cannot easily associate predicted depths with their spatial origin in the image.

### Encoder–Decoder strengths

| Property | Benefit |
|----------|---------|
| **Multi-scale representation** | Encoder captures features at all scales; decoder uses them all |
| **Skip connections** | Preserve high-frequency details (edges, corners) that bottleneck layers lose |
| **Pretrained encoder** | ImageNet pretraining gives free semantic understanding (chairs, walls, floors) |
| **Progressive upsampling** | Gradual resolution recovery produces smoother, more consistent depth maps |
| **Symmetric design** | Decoder mirrors encoder, making the information flow natural and learnable |

### Comparison with alternatives

| Architecture | RMSE (NYU) | Parameters | Notes |
|-------------|-----------|-----------|-------|
| Plain CNN | ~0.85 m | Small | No spatial awareness |
| Encoder only + resize | ~0.72 m | Medium | Blurry output |
| **Encoder–Decoder (ours)** | **~0.55 m** | **~25M** | Good balance |
| Transformer-based (e.g., DPT) | ~0.35 m | ~120M | State of the art, expensive |

---

## 14. Future Improvements

### Short term (easy wins)

- [ ] **Larger backbone**: Replace ResNet-34 with EfficientNet-B5 or DenseNet-161 for better feature extraction
- [ ] **Larger input resolution**: Train at 480×640 (full NYU resolution) with larger batch on multi-GPU
- [ ] **Data augmentation**: Add mixup, CutMix, and more aggressive colour jitter
- [ ] **Model ensemble**: Average predictions from multiple checkpoints

### Medium term (significant improvements)

- [ ] **Attention mechanism**: Add channel-wise (SE blocks) or spatial attention to the decoder
- [ ] **BTS (Big-to-Small)**: Implement the local planar guidance layers from Lee et al. (2019)
- [ ] **Depth completion**: Fuse sparse LiDAR hints with monocular predictions
- [ ] **Uncertainty estimation**: Predict aleatoric uncertainty alongside depth (use dropout or ensemble)

### Long term (research directions)

- [ ] **Vision Transformer encoder**: Replace ResNet with ViT-L or Swin Transformer for global context
- [ ] **Self-supervised training**: Use ego-motion between video frames as free supervision (Monodepth2)
- [ ] **Generalisation**: Train on mixed datasets (NYU + KITTI + ScanNet) for outdoor/indoor generalisation
- [ ] **3D reconstruction**: Use predicted depth maps to build a 3D point cloud with Open3D
- [ ] **Real-time inference**: Quantise the model (INT8) and deploy with TensorRT for <30ms inference

---

## 15. References

1. **Eigen et al.** (2014). *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*. NeurIPS 2014. [arXiv:1406.2283](https://arxiv.org/abs/1406.2283)

2. **Ronneberger et al.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

3. **He et al.** (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

4. **Lee et al.** (2019). *From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation*. [arXiv:1907.10326](https://arxiv.org/abs/1907.10326)

5. **Godard et al.** (2019). *Digging Into Self-Supervised Monocular Depth Estimation*. ICCV 2019. [arXiv:1806.01260](https://arxiv.org/abs/1806.01260)

6. **Ranftl et al.** (2021). *Vision Transformers for Dense Prediction*. ICCV 2021. [arXiv:2103.13413](https://arxiv.org/abs/2103.13413)

7. **NYU Depth V2**: Silberman et al. (2012). *Indoor Segmentation and Support Inference from RGBD Images*. ECCV 2012.

---

## License

This project is for educational purposes. The NYU Depth V2 dataset is subject to its own license terms.
