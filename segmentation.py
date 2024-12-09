from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")
results = model("images/2022_12_21_14_12_IMG_6239.JPG")
results[0].show()