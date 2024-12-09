import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Set device to CUDA, MPS, or CPU based on availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configurations for DepthAnything
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Choose the encoder type
encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

# Load DepthAnything model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Load image for depth estimation
raw_img = cv2.imread("images/1.JPG")

# Generate depth map
depth_map = model.infer_image(raw_img)  # Depth map (HxW numpy array)

# Save depth map output
cv2.imwrite('output/example_depth.png', depth_map)
