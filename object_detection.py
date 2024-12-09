from ultralytics import YOLO
import cv2

# Load YOLOv5 model (using a pre-trained weights file)
model = YOLO("yolov8n.pt")  # Example, replace with the actual weights path

# Load image
image_path = "images/1.JPG"
img = cv2.imread(image_path)

# Perform object detection
results = model(img)

# Display or save the results
results[0].show()  # Shows the image with detections
results.save(save_dir="output")  # Saves output images to the 'output' folder
