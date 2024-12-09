import streamlit as st
import cv2
import torch
import numpy as np
import os
import matplotlib
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# Function to load DepthAnything model
@st.cache_resource
def load_depth_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    encoder = 'vitl'  # Choose the encoder type
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model

# Function to load YOLO model for detection and segmentation
@st.cache_resource
def load_yolo_model(model_type='yolo11n.pt'):
    return YOLO(model_type)

# Function to process the uploaded image for depth estimation
def process_depth_estimation(model, image_path):
    raw_img = cv2.imread(image_path)
    depth_map = model.infer_image(raw_img)
    # Normalize depth values to 0-255
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

# Use 'jet' colormap for warm-to-cool visualization
    cmap = matplotlib.cm.get_cmap('jet')  # 'jet' maps warm colors to closer objects
    depth_color = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB format

    return depth_color

# Function to process YOLO object detection
def process_detection(model, image_path):
    img = cv2.imread(image_path)
    results = model(img)
    return results[0].plot()

# Function to process YOLO segmentation
def process_segmentation(model, image_path):
    img = cv2.imread(image_path)
    results = model(image_path)
    return results[0].plot()

# Function to combine the results
def combine_results(seg_img, det_img, depth_img):
    split_region = np.ones((seg_img.shape[0], 50, 3), dtype=np.uint8) * 255
    final_result = cv2.hconcat([seg_img, split_region, det_img, split_region, depth_img])
    return final_result

# Streamlit App Interface
st.title("Image Processing App")
st.sidebar.title("Choose a Category")
category = st.sidebar.radio(
    "Select a task:",
    ("Depth Estimation", "Object Detection (YOLO)", "Segmentation (YOLO)", "Combined Output")
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    img_path = f"temp_{uploaded_file.name}"
    img.save(img_path)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Load the models once
    depth_model = load_depth_model()
    det_model = load_yolo_model("yolo11n.pt")
    seg_model = load_yolo_model("yolo11n-seg.pt")

    if category == "Depth Estimation":
        st.subheader("Depth Estimation Results")
        with st.spinner("Processing..."):
            depth_img = process_depth_estimation(depth_model, img_path)
            st.image(depth_img, caption="Depth Map", use_container_width=True)
            st.success("Depth Estimation Completed!")

    elif category == "Object Detection (YOLO)":
        st.subheader("YOLO Object Detection Results")
        with st.spinner("Processing..."):
            det_img = process_detection(det_model, img_path)
            st.image(det_img, caption="Detection Results", use_container_width=True)
            st.success("Object Detection Completed!")

    elif category == "Segmentation (YOLO)":
        st.subheader("YOLO Segmentation Results")
        with st.spinner("Processing..."):
            seg_img = process_segmentation(seg_model, img_path)
            st.image(seg_img, caption="Segmentation Results", use_container_width=True)
            st.success("Segmentation Completed!")

    elif category == "Combined Output":
        st.subheader("Combined Results: Depth Estimation + Segmentation + Detection")
        with st.spinner("Processing..."):
            seg_img = process_segmentation(seg_model, img_path)
            det_img = process_detection(det_model, img_path)
            depth_img = process_depth_estimation(depth_model, img_path)
            final_result = combine_results(seg_img, det_img, depth_img)
            st.image(final_result, caption="Combined Output", use_container_width=True)
            st.success("Combined Task Completed!")
