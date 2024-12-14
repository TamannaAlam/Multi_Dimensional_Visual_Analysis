# Multi-Dimensional Visual Analysis
This repository showcases a multi-modal approach for visual analysis using cutting-edge tools for depth estimation, segmentation, and object detection.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Depth Map Estimation (Depth Anything V2)](#depth-map-estimation-depth-anything-v2)
   - [Segmentation (Segment Anything)](#segmentation-segment-anything)
   - [Fast Lightweight Segmentation (FastSAM)](#fast-lightweight-segmentation-fastsam)
   - [Object Detection (YOLOv8)](#object-detection-yolov8)
6. [Model Weights](#model-weights)
7. [Outputs](#outputs)
8. [Future Work](#future-work)
9. [Acknowledgments](#acknowledgments)

## 1. Overview
This project integrates the following tools for visual analysis:

- **Depth Anything V2**: Generate accurate depth maps.
- **Segment Anything (SAM)**: Powerful segmentation model for various objects.
- **FastSAM**: Lightweight and fast segmentation alternative.
- **YOLOv8**: Real-time object detection and segmentation.

## 2. Features
- **Depth Map Generation**: Analyze depth information from input images.
- **Object Segmentation**: Mask and identify objects using advanced segmentation models.
- **Fast Segmentation**: Lightweight segmentation for speed-optimized workflows.
- **Object Detection**: Detect and classify objects in an image using YOLOv8.

## 3. Requirements
- Python >= 3.8
- CUDA (optional, for GPU support)

### Libraries:
- PyTorch
- OpenCV
- Ultralytics
- FastSAM
- Segment Anything

## 4. Installation
Follow these steps to set up the project:

### Step 1: Clone the Repository

```bash
git clone https://github.com/TamannaAlam/Multi_Dimensional_Visual_Analysis.git
cd Multi_Dimensional_Visual_Analysis
