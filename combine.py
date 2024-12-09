import cv2
import torch
import numpy as np
import os
import matplotlib
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# Set the input image path
image_path = "images/03.jpg"

# -----------------------------
# 1. YOLO Segmentation
# -----------------------------
seg_model = YOLO("yolo11n-seg.pt") 
seg_results = seg_model(image_path)
# seg_results[0].plot() returns a numpy array with segmentation overlay
seg_img = seg_results[0].plot()

# -----------------------------
# 2. YOLO Detection
# -----------------------------
det_model = YOLO("yolo11n.pt") 
det_results = det_model(image_path)
# det_results[0].plot() returns a numpy array with bounding box overlays
det_img = det_results[0].plot()

# -----------------------------
# 3. Depth Estimation
# -----------------------------
DEVICE = 'cpu' 

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

depth_anything = DepthAnythingV2(**model_configs['vits'])
depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

raw_image = cv2.imread(image_path)
input_size = 518
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

depth = depth_anything.infer_image(raw_image, input_size)
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
# Apply a colormap for depth visualization
depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

# -----------------------------
# Combine the three results
# -----------------------------
# We will place them side-by-side with a white spacer between each.
split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255

# Concatenate horizontally: segmentation | spacer | detection | spacer | depth
final_result = cv2.hconcat([seg_img, split_region, det_img, split_region, depth_color])

# -----------------------------
# Save the final combined image
# -----------------------------
os.makedirs("final_output", exist_ok=True)
output_path = os.path.join("final_output", "combined_results.png")
cv2.imwrite(output_path, final_result)

print(f"Combined result saved at {output_path}")
