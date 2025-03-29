import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_dir = 'brain_tumor_dataset'  # Directory containing 'train' and 'test' folders
model_path = os.path.join('models', 'brain_tumor_detection_model2.h5')

# Load the saved model
model = load_model(model_path)

# Define preprocessing for testing
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'test'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False  # Do not shuffle for consistent evaluation
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on test data
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

# Print predictions
for img_path, pred in zip(test_generator.filepaths, predicted_labels):
    print(f"Image: {img_path}, Predicted: {'Tumor' if pred == 1 else 'No Tumor'}")