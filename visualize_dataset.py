import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("dataset.npz", allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]
categories = data["categories"]

# No need to use argmax here
y_train_indices = y_train

# Show 5 random images per category
plt.figure(figsize=(15, 10))
images_shown = {i: 0 for i in range(len(categories))}
num_images = 20
i = 0
idx = 0
while i < num_images and idx < len(X_train):
    label = y_train_indices[idx]
    if images_shown[label] < 5:
        plt.subplot(4, 5, i + 1)
        plt.imshow(X_train[idx].reshape(256, 256), cmap="gray")
        plt.title(categories[label])
        plt.axis("off")
        images_shown[label] += 1
        i += 1
    idx += 1

plt.suptitle("📸 Sample Images from Each Category", fontsize=18)
plt.tight_layout()
plt.show()
