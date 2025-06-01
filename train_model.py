import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical

# Load dataset
data = np.load("dataset.npz", allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
categories = data["categories"]

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=len(categories))
y_test_cat = to_categorical(y_test, num_classes=len(categories))

# Build CNN model
model = Sequential([
    Input(shape=(256, 256, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test, y_test_cat))

# Save model
model.save("vehicle_model.h5")
print("✅ Model trained and saved as vehicle_model.h5")
