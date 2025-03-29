import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil

# Define paths
dataset_dir = 'brain_tumor_dataset'  # Root folder containing 'yes' and 'no' subfolders

# Load image paths and labels
image_paths = []
labels = []

# Load images from 'yes' folder (tumor)
yes_dir = os.path.join(dataset_dir, 'yes')
for img_name in os.listdir(yes_dir):
    img_path = os.path.join(yes_dir, img_name)
    image_paths.append(img_path)
    labels.append(1)  # 1 for tumor

# Load images from 'no' folder (no tumor)
no_dir = os.path.join(dataset_dir, 'no')
for img_name in os.listdir(no_dir):
    img_path = os.path.join(no_dir, img_name)
    image_paths.append(img_path)
    labels.append(0)  # 0 for no tumor

# Convert to NumPy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

# Function to save split data to disk
def save_split_data(image_paths, labels, output_dir):
    for img_path, label in zip(image_paths, labels):
        class_name = 'yes' if label == 1 else 'no'
        dest_dir = os.path.join(output_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# Save training and testing data
save_split_data(X_train_paths, y_train, os.path.join(dataset_dir, 'train'))
save_split_data(X_test_paths, y_test, os.path.join(dataset_dir, 'test'))

# Define data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define preprocessing for testing (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create generators
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'train'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'test'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add callbacks to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'brain_tumor_detection_model2.h5'))

# Plot training results
os.makedirs('results', exist_ok=True)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join('results', 'accuracy_plot.png'))
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join('results', 'loss_plot.png'))
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")