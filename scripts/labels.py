import os
import pandas as pd

# Define paths
train_dir = os.path.join('brain_tumor_dataset', 'Training', 'glioma')
test_dir = os.path.join('brain_tumor_dataset', 'Testing', 'glioma')

# Function to generate labels
def generate_labels(image_dir, label):
    image_paths = []
    labels = []
    for file in os.listdir(image_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            # Get the full path to the image
            img_path = os.path.join(image_dir, file)
            # Get the relative path (e.g., Training/glioma/Tr-gl_0010.jpg)
            rel_path = os.path.relpath(img_path, 'brain_tumor_dataset')
            image_paths.append(rel_path)
            labels.append(label)
    return pd.DataFrame({'Image': image_paths, 'Label': labels})

# Generate labels for training and testing data
train_labels = generate_labels(train_dir, label=1)  # Label 1 for training images (tumor)
test_labels = generate_labels(test_dir, label=0)    # Label 0 for testing images (non-tumor)

# Combine into one DataFrame
labels_df = pd.concat([train_labels, test_labels])

# Save to CSV
labels_df.to_csv(os.path.join('brain_tumor_dataset', 'labels.csv'), index=False)

print("labels.csv generated successfully!")