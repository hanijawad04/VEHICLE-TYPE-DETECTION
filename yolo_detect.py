# yolo_detect.py
import os
from ultralytics import YOLO
import cv2

# Load YOLOv8 model (downloads if not present)
model = YOLO("yolov8m.pt")

# Define allowed vehicle types (depends on COCO dataset)
vehicle_labels = ["car", "motorcycle", "bus", "truck", "bicycle"]

def detect_vehicles(image_path, output_path):
    results = model(image_path)[0]  # Get first prediction result
    detections = []

    img = cv2.imread(image_path)

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if label in vehicle_labels:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((label, conf, (x1, y1, x2, y2)))
            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save output image with boxes
    cv2.imwrite(output_path, img)
    return detections
