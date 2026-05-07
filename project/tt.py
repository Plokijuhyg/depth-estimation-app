import csv
import numpy as np
from PIL import Image

# تحقق من أول 5 train depth images
with open("dataset/nyu2_train.csv") as f:
    rows = list(csv.reader(f))[:5]

print("=== TRAIN DEPTH ===")
for row in rows:
    path = "dataset/" + row[1].replace("data/", "")
    img  = Image.open(path).convert("RGB")
    arr  = np.array(img, dtype=np.float32).mean(axis=-1) / 255.0
    depth = 1000.0 / np.clip(arr * 1000.0, 10.0, 1000.0)
    print(f"  min={depth.min():.2f} max={depth.max():.2f} mean={depth.mean():.2f}")

# تحقق من أول 5 val depth images
with open("dataset/nyu2_test.csv") as f:
    rows = list(csv.reader(f))[:5]

print("\n=== VAL DEPTH ===")
for row in rows:
    path = "dataset/" + row[1].replace("data/", "")
    img  = Image.open(path)
    arr  = np.array(img, dtype=np.float32) / 1000.0
    print(f"  min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f}")