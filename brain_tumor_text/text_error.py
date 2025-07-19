import pandas as pd
import numpy as np

def introduce_errors(data, error_rate=0.1):
    # Randomly flip the tumor labels to introduce mislabeling
    num_samples = len(data)
    num_errors = int(num_samples * error_rate)
    error_indices = np.random.choice(num_samples, num_errors, replace=False)
    
    # Flip tumor labels (tumor_present) randomly
    for idx in error_indices:
        data.loc[idx, 'tumor_present'] = 1 - data.loc[idx, 'tumor_present']
        
    # Introduce inconsistent data: flip symptoms randomly for error cases
    symptom_errors = [
        "Unrelated symptom A", "Unrelated symptom B", "Random symptom", "No symptoms"
    ]
    
    for idx in error_indices:
        data.loc[idx, 'symptoms'] = np.random.choice(symptom_errors)
    
    # Add extreme outliers in features like age or symptom duration
    outlier_indices = np.random.choice(num_samples, num_errors, replace=False)
    for idx in outlier_indices:
        # Add extreme values for age or symptom duration
        if np.random.rand() > 0.5:
            data.loc[idx, 'age'] = np.random.choice([100, 110, 120])  # Outlier ages
        else:
            data.loc[idx, 'symptom_duration_weeks'] = np.random.choice([100, 200, 300])  # Outlier duration
    
    return data

# Load the dataset
data = pd.read_csv('brain_tumor_text_dataset.csv')

# Introduce errors into the dataset (e.g., 10% errors)
data_with_errors = introduce_errors(data, error_rate=0.1)

# Save the dataset with errors
data_with_errors.to_csv('brain_tumor_text_dataset2.csv', index=False)
print("Dataset with errors created and saved as 'brain_tumor_text_dataset2.csv'")
