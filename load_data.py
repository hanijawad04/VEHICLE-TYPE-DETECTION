import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = "processed_images"  # Make sure this has folders: car, bike, bus, truck
categories = sorted(os.listdir(data_dir))  # Sorted to keep label order consistent

X = []
y = []

print("📦 Loading images from folders...")

for idx, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):
        file_path = os.path.join(category_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {file_path}")
            continue

        img = cv2.resize(img, (256, 256))
        X.append(img)
        y.append(idx)  # Correct label index based on category

X = np.array(X).reshape(-1, 256, 256, 1) / 255.0  # Normalize
y = np.array(y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save everything, including categories
np.savez("dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, categories=np.array(categories))

print("✅ Dataset created and saved as dataset.npz")
print("🧾 Categories:", categories)
