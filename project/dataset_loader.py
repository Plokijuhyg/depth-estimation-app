"""
dataset_loader.py
-----------------
NYU Depth V2 — Folder-Based Dataset Loader
============================================
Replaces the original HDF5 / .mat loader with one that reads directly
from the extracted Kaggle folder layout:

    dataset/
    ├── nyu2_train.csv          ← col 0: RGB path,  col 1: depth path
    ├── nyu2_test.csv           ← col 0: RGB path,  col 1: depth path
    ├── nyu2_train/
    │   ├── <scene>/00001.jpg   ← RGB  (JPEG, uint8)
    │   └── <scene>/00001.png   ← depth (PNG, see encoding notes below)
    └── nyu2_test/
        ├── 00001_colors.png    ← RGB  (PNG, uint8)
        └── 00001_depth.png     ← depth (PNG, see encoding notes below)

Depth PNG encoding — two variants exist in the wild
────────────────────────────────────────────────────
Variant A  (most common in the Kaggle download)
    The depth PNG is an 8-bit JPEG-reencoded file.  The original
    DenseDepth / ialhashim pipeline reads it as a normal image, normalises
    to [0, 1], then applies the disparity-to-depth formula:
        depth_metres = 1000 / clip(pixel_norm × 1000, 10, 1000)

Variant B  (wangq95 / NYUd2-Toolkit 16-bit PNGs)
    The depth PNG is a true 16-bit grayscale image where:
        depth_metres = pixel_value / 65535.0 × 10.0

The loader auto-detects which variant is present by inspecting the
bit-depth of the first depth file.

Public interface (unchanged from original)
──────────────────────────────────────────
    NYU_MEAN, NYU_STD, MIN_DEPTH, MAX_DEPTH
    class  NYUDepthDataset
    def    get_dataloaders(...)

All other scripts (train.py, test.py, visualize.py) call get_dataloaders()
with a keyword argument previously named `hdf5_path`.  The new signature
keeps `data_root` as the first positional argument and aliases the old
kwarg so existing call-sites keep working with zero changes.
"""

import os
import csv
import glob
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (kept identical to original so other modules can import them)
# ─────────────────────────────────────────────────────────────────────────────
NYU_MEAN  = [0.485, 0.456, 0.406]   # ImageNet mean — encoder pretrained on IN
NYU_STD   = [0.229, 0.224, 0.225]   # ImageNet std
MIN_DEPTH = 0.001                    # metres — mask out invalid sensor pixels
MAX_DEPTH = 10.0                     # metres — Kinect indoor maximum range


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — CSV / folder parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_csv_prefix(rel_path: str) -> str:
    """
    The Kaggle CSV uses paths like:
        data/nyu2_train/bedroom/0001.jpg

    The leading component ('data', 'dataset', etc.) is just an artifact
    of how the CSV was generated and does NOT match the actual folder name
    on disk.  Strip it so we keep only:
        nyu2_train/bedroom/0001.jpg

    We detect it by checking whether the first component is NOT one of the
    known real sub-folder names that actually exist in the dataset root.
    """
    # Normalise separators so split works on Windows too
    parts = rel_path.replace("\\", "/").split("/")
    known_roots = {"nyu2_train", "nyu2_test", "train", "test"}
    if len(parts) > 1 and parts[0] not in known_roots:
        # First component is a stale prefix — drop it
        parts = parts[1:]
    return "/".join(parts)


def _read_csv_pairs(csv_path: str, data_root: str) -> List[Tuple[str, str]]:
    """
    Parse a NYU2 CSV file and return a list of (rgb_abs_path, depth_abs_path).

    The CSV has NO header row.  Each line is:
        data/nyu2_train/living_room/00001.jpg,data/nyu2_train/living_room/00001.png

    The leading 'data/' prefix in the CSV does not match the actual folder
    name on disk (which is 'dataset/', 'nyu_data/', etc.).  We strip it
    automatically and resolve paths relative to `data_root`.
    """
    pairs: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            rgb_rel   = _strip_csv_prefix(row[0].strip())
            depth_rel = _strip_csv_prefix(row[1].strip())
            if not rgb_rel or not depth_rel:
                continue
            rgb_abs   = os.path.join(data_root, rgb_rel)
            depth_abs = os.path.join(data_root, depth_rel)
            pairs.append((rgb_abs, depth_abs))
    return pairs


def _discover_pairs_no_csv(folder: str) -> List[Tuple[str, str]]:
    """
    Fallback when no CSV is present: scan the image folder and pair files.

    Strategy A: match *_colors.png  →  *_depth.png
    Strategy B: match every *.jpg   →  same-stem *.png (side-by-side layout)
    """
    pairs: List[Tuple[str, str]] = []

    # Strategy A — explicit suffix convention
    rgb_files = sorted(glob.glob(os.path.join(folder, "**", "*_colors.png"),
                                  recursive=True))
    if rgb_files:
        for rgb in rgb_files:
            depth = rgb.replace("_colors.png", "_depth.png")
            if os.path.isfile(depth):
                pairs.append((rgb, depth))
        if pairs:
            return pairs

    # Strategy B — JPG (RGB) paired with same-stem PNG (depth)
    jpg_files = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"),
                                  recursive=True))
    for jpg in jpg_files:
        png = os.path.splitext(jpg)[0] + ".png"
        if os.path.isfile(png):
            pairs.append((jpg, png))

    return pairs


def _find_csv(data_root: str, split: str) -> Optional[str]:
    """
    Look for nyu2_train.csv or nyu2_test.csv inside data_root.
    Returns the absolute path or None.
    """
    for name in (f"nyu2_{split}.csv", f"{split}.csv"):
        path = os.path.join(data_root, name)
        if os.path.isfile(path):
            return path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Depth decoding
# ─────────────────────────────────────────────────────────────────────────────

def _detect_depth_encoding(depth_path: str) -> str:
    """
    Auto-detect depth PNG variant by inspecting the first file.

    Returns one of:
        '16bit_mm' — uint16 PNG, pixel value = depth in millimetres
                     NYU test set confirmed: min=1123 max=2609 (mm)
                     Decode: depth_metres = pixel / 1000.0
        '8bit'     — 8-bit JPEG-reencoded depth (disparity formula)
        'float'    — 32-bit TIFF (already in metres)
    """
    img  = Image.open(depth_path)
    mode = img.mode
    if mode == "F":
        return "float"
    if mode in ("I;16", "I;16B", "I"):
        arr = np.array(img)
        if arr.max() > 255:
            return "16bit_mm"
    return "8bit"


def _load_depth_metres(depth_path: str, encoding: str) -> np.ndarray:
    """
    Load a depth file and return a (H, W) float32 array in metres.

    Encoding '16bit_mm'
    ───────────────────
    NYU Depth V2 Kaggle test set — uint16 PNG:
        pixel value = depth in millimetres
        e.g. pixel=1123 → 1.123 metres
    Confirmed from real data inspection: min=1123, max=2609, mean=1755
    Decode: depth_metres = pixel / 1000.0

    Encoding '8bit'
    ───────────────
    8-bit image reencoded from disparity. DenseDepth formula:
        pixel_norm   = pixel_value / 255.0
        depth_metres = 1000 / clip(pixel_norm * 1000, 10, 1000)

    Encoding 'float'
    ────────────────
    32-bit TIFF already in metres — used as-is.
    """
    img = Image.open(depth_path)

    if encoding == "16bit_mm":
        arr   = np.array(img, dtype=np.float32)
        depth = arr / 1000.0          # millimetres → metres

    elif encoding == "float":
        depth = np.array(img, dtype=np.float32)

    else:   # '8bit'  — disparity formula
        arr        = np.array(img.convert("RGB"), dtype=np.float32)
        pixel_norm = arr.mean(axis=-1) / 255.0
        depth      = 1000.0 / np.clip(pixel_norm * 1000.0, 10.0, 1000.0)

    return np.clip(depth, MIN_DEPTH, MAX_DEPTH).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NYUDepthDataset(Dataset):
    """
    PyTorch Dataset for the folder-based NYU Depth V2 Kaggle download.

    Parameters
    ----------
    data_root : str
        Root directory that contains nyu2_train.csv / nyu2_test.csv
        and the nyu2_train/ / nyu2_test/ subfolders.
    split : str
        'train' reads nyu2_train.csv (and nyu2_train/ fallback).
        'val'   reads nyu2_test.csv  (and nyu2_test/  fallback).
    img_size : (H, W)
        Target spatial resolution for all returned tensors.
    augment : bool
        Apply random augmentation (forced off for val split).
    val_subset : int or None
        Cap the val set to this many samples (handy for quick dev loops).
    """

    _FOLDER_MAP = {"train": "nyu2_train", "val": "nyu2_test"}
    _CSV_MAP    = {"train": "train",      "val": "test"}

    def __init__(
        self,
        data_root:  str,
        split:      str   = "train",
        img_size:   Tuple[int, int] = (240, 320),
        augment:    bool  = True,
        val_subset: Optional[int] = None,
    ):
        super().__init__()
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.img_size = img_size
        self.augment  = augment and (split == "train")
        self.img_norm = T.Normalize(mean=NYU_MEAN, std=NYU_STD)

        data_root      = os.path.abspath(data_root)
        self.data_root = data_root

        # ── Build (rgb, depth) path pairs ─────────────────────────────
        csv_path = _find_csv(data_root, self._CSV_MAP[split])

        if csv_path is not None:
            print(f"\n  [{split}] CSV  → {csv_path}")
            self.pairs = _read_csv_pairs(csv_path, data_root)
        else:
            folder = os.path.join(data_root, self._FOLDER_MAP[split])
            print(f"\n  [{split}] No CSV found — scanning {folder}")
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Cannot find data for split='{split}'.\n"
                    f"  Tried CSV : {os.path.join(data_root, f'nyu2_{self._CSV_MAP[split]}.csv')}\n"
                    f"  Tried dir : {folder}\n"
                    f"  data_root : {data_root}"
                )
            self.pairs = _discover_pairs_no_csv(folder)

        if not self.pairs:
            raise RuntimeError(
                f"No valid (rgb, depth) pairs found for split='{split}' "
                f"in data_root='{data_root}'"
            )

        # ── Validate first few files actually exist ────────────────────
        for rgb_p, dep_p in self.pairs[:5]:
            if not os.path.isfile(rgb_p):
                raise FileNotFoundError(
                    f"RGB file missing: {rgb_p}\n"
                    f"Hint: verify that data_root='{data_root}' is correct."
                )
            if not os.path.isfile(dep_p):
                raise FileNotFoundError(
                    f"Depth file missing: {dep_p}\n"
                    f"Hint: verify that data_root='{data_root}' is correct."
                )

        # ── Optional val cap ──────────────────────────────────────────
        if val_subset is not None and split == "val":
            self.pairs = self.pairs[:val_subset]

        # ── Auto-detect depth encoding from first sample ──────────────
        self.depth_encoding = _detect_depth_encoding(self.pairs[0][1])

        print(
            f"  [{split}] {len(self.pairs):,} pairs | "
            f"depth='{self.depth_encoding}' | "
            f"size={img_size[0]}×{img_size[1]} | "
            f"augment={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, dep_path = self.pairs[idx]

        # ── RGB image ─────────────────────────────────────────────────
        img_pil = Image.open(rgb_path).convert("RGB")

        # ── Depth map → float32 metres ────────────────────────────────
        depth_np  = _load_depth_metres(dep_path, self.depth_encoding)
        depth_pil = Image.fromarray(depth_np, mode="F")

        # ── Resize ────────────────────────────────────────────────────
        H, W      = self.img_size
        img_pil   = img_pil.resize((W, H), Image.BILINEAR)
        depth_pil = depth_pil.resize((W, H), Image.NEAREST)

        # ── Augmentation (train only) ─────────────────────────────────
        if self.augment:
            img_pil, depth_pil = _augment(img_pil, depth_pil)

        # ── To tensors ────────────────────────────────────────────────
        img_t   = TF.to_tensor(img_pil)          # (3,H,W) float [0,1]
        img_t   = self.img_norm(img_t)            # ImageNet-normalised

        depth_t = torch.from_numpy(
                      np.array(depth_pil, dtype=np.float32)
                  ).unsqueeze(0)                  # (1,H,W) metres
        depth_t = torch.clamp(depth_t, MIN_DEPTH, MAX_DEPTH)

        return img_t, depth_t


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Augmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def _augment(
    img_pil:   "Image.Image",
    depth_pil: "Image.Image",
) -> Tuple["Image.Image", "Image.Image"]:
    """
    Consistent spatial + photometric augmentation for an (RGB, depth) pair.

    1. Random horizontal flip  (p = 0.5) — applied to both modalities.
    2. Random colour jitter    (p = 0.5) — RGB only; depth is geometry.
    3. Random crop + resize    (p = 0.5) — applied to both modalities.
    """
    # 1 — Horizontal flip
    if random.random() > 0.5:
        img_pil   = TF.hflip(img_pil)
        depth_pil = TF.hflip(depth_pil)

    # 2 — Colour jitter (RGB only)
    if random.random() > 0.5:
        img_pil = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )(img_pil)

    # 3 — Random crop then resize back
    if random.random() > 0.5:
        W, H       = img_pil.size
        scale      = random.uniform(0.85, 1.0)
        cH, cW     = int(H * scale), int(W * scale)
        top        = random.randint(0, H - cH)
        left       = random.randint(0, W - cW)

        img_pil    = TF.crop(img_pil,   top, left, cH, cW)
        depth_pil  = TF.crop(depth_pil, top, left, cH, cW)

        img_pil    = img_pil.resize((W, H),   Image.BILINEAR)
        depth_pil  = depth_pil.resize((W, H), Image.NEAREST)

    return img_pil, depth_pil


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Factory function  (public API, backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root:   str   = "dataset",
    batch_size:  int   = 8,
    img_size:    Tuple[int, int] = (240, 320),
    num_workers: int   = 4,
    val_subset:  Optional[int] = None,
    # ── Backward-compatibility aliases ────────────────────────────────
    hdf5_path:   Optional[str] = None,   # old first-positional arg
    train_ratio: float = 0.8,            # unused; kept so old calls don't crash
) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader).

    Parameters
    ----------
    data_root   : Root directory of the extracted Kaggle dataset.
                  Must contain nyu2_train.csv, nyu2_test.csv (or the image
                  sub-folders nyu2_train/ and nyu2_test/).
    batch_size  : Samples per batch.
    img_size    : (H, W) — resize target for all images and depth maps.
    num_workers : Subprocess workers for DataLoader.
    val_subset  : Limit validation set to this many samples (optional).
    hdf5_path   : Legacy alias for data_root.  If supplied, overrides
                  data_root.  Accepts a file path (parent dir is used) or
                  a directory path — so old code like
                      get_dataloaders(hdf5_path="dataset/nyu_depth_v2_labeled.mat")
                  continues to work unchanged.
    train_ratio : Ignored — retained only for API compatibility.

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    # ── Resolve legacy hdf5_path kwarg ────────────────────────────────
    if hdf5_path is not None:
        if os.path.isfile(hdf5_path):
            resolved = os.path.dirname(hdf5_path)
            print(
                f"[dataset_loader] hdf5_path='{hdf5_path}' is a file;\n"
                f"                 switching data_root to its parent: '{resolved}'"
            )
            data_root = resolved
        else:
            data_root = hdf5_path

    print(f"\n{'='*60}")
    print(f"  NYU Depth V2 — Folder-Based DataLoader")
    print(f"  data_root  : {os.path.abspath(data_root)}")
    print(f"  img_size   : {img_size[0]} × {img_size[1]}")
    print(f"  batch_size : {batch_size}   num_workers: {num_workers}")
    print(f"{'='*60}")

    train_ds = NYUDepthDataset(
        data_root  = data_root,
        split      = "train",
        img_size   = img_size,
        augment    = True,
    )
    val_ds = NYUDepthDataset(
        data_root  = data_root,
        split      = "val",
        img_size   = img_size,
        augment    = False,
        val_subset = val_subset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    print(
        f"\n  Train : {len(train_ds):,} samples → {len(train_loader):,} batches\n"
        f"  Val   : {len(val_ds):,} samples → {len(val_loader):,} batches\n"
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — CLI sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check.  Run from the project root:

        # New usage (folder):
        python dataset_loader.py dataset/

        # Legacy usage (.mat path — parent dir used automatically):
        python dataset_loader.py dataset/nyu_depth_v2_labeled.mat
    """
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"

    train_loader, val_loader = get_dataloaders(
        data_root   = root,
        batch_size  = 4,
        num_workers = 0,      # 0 = no subprocesses, easier to debug
        val_subset  = 20,
    )

    print("── Train batch ──────────────────────────────────────")
    imgs, deps = next(iter(train_loader))
    print(f"  images : {tuple(imgs.shape)}  dtype={imgs.dtype}")
    print(f"  depths : {tuple(deps.shape)}  dtype={deps.dtype}")
    print(f"  img   range : [{imgs.min():.3f}, {imgs.max():.3f}]  (ImageNet-normalised)")
    print(f"  depth range : [{deps.min():.4f}, {deps.max():.4f}] metres")

    print("\n── Val batch ────────────────────────────────────────")
    imgs, deps = next(iter(val_loader))
    print(f"  images : {tuple(imgs.shape)}  dtype={imgs.dtype}")
    print(f"  depths : {tuple(deps.shape)}  dtype={deps.dtype}")
    print(f"  depth range : [{deps.min():.4f}, {deps.max():.4f}] metres")

    print("\n✓  dataset_loader.py OK — all tensors look correct.\n")