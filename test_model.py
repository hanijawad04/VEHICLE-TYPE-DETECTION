# test_model.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
try:
    model = load_model("vehicle_model.h5")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define your categories (ensure order matches training)
categories = ["bike", "bus", "car", "truck"]  # Adjust this based on your folder order

# Test image path (change as needed)
img_path = "test_images/car1.jpeg"

try:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image at {img_path} not found.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 256, 256, 1)

    prediction = model.predict(img)
    predicted_label = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"✅ Predicted Vehicle Type: {predicted_label} ({confidence:.2f}% confidence)")

except Exception as e:
    print(f"❌ Error processing image: {e}")
