from ultralytics import YOLO

# Load the YOLOv8n model (nano version for fast inference)
model = YOLO("yolov8m.pt")

# Confirm model structure
model.info()
