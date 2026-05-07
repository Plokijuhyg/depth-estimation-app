import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load MiDaS Model
# =========================
model_type = "DPT_Hybrid"  # الأفضل لمشروعك

model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# Load transforms
# =========================
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# =========================
# Load image
# =========================
img_path = "../dataset/nyu2_test/00013_colors.png"  # <-- حط صورتك هون

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# Preprocess
# =========================
input_batch = transform(img).to(device)

# =========================
# Prediction
# =========================
with torch.no_grad():
    prediction = model(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# =========================
# Normalize for visualization
# =========================
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# =========================
# Show results
# =========================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Depth Map (MiDaS)")
plt.imshow(depth_map, cmap="inferno")
plt.axis("off")

plt.show()

# =========================
# Save output
# =========================
cv2.imwrite("depth_output.png", (depth_map * 255).astype(np.uint8))

print("✅ Done! Depth map saved as depth_output.png")