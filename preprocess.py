# preprocess.py
import cv2
import os
import numpy as np

input_folder = "label/"
output_folder = "processed_images/"
os.makedirs(output_folder, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    output_category_path = os.path.join(output_folder, category)
    os.makedirs(output_category_path, exist_ok=True)

    for filename in os.listdir(category_path):
        if not filename.lower().endswith(image_extensions):
            continue

        img_path = os.path.join(category_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Skipping unreadable file: {img_path}")
            continue

        img = cv2.resize(img, (256, 256))
        img = np.uint8(img)

        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_category_path, f"{base_name}.jpg")
        cv2.imwrite(save_path, img)

print("✅ Image Preprocessing Complete — no augmentations applied.")
