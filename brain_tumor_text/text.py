import pandas as pd
import numpy as np

def create_brain_tumor_dataset(num_samples=500):
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, num_samples+1),
        'age': np.random.randint(15, 85, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.48, 0.52]),
        'symptoms': [],
        'symptom_duration_weeks': np.random.randint(1, 52, num_samples),
        'family_history': np.random.choice(['Yes', 'No'], num_samples, p=[0.2, 0.8]),
        'medical_history': [],
        'neurological_exam': [],
        'imaging_report': [],
        'lab_results': [],
        'tumor_present': []
    }
    
    # Possible values for each field
    tumor_symptoms = [
        "Persistent headaches worse in morning",
        "Nausea and vomiting",
        "Blurred or double vision",
        "Seizures",
        "Difficulty speaking",
        "Memory problems",
        "Loss of balance",
        "Weakness in limbs",
        "Personality changes",
        "Hearing problems"
    ]
    
    non_tumor_symptoms = [
        "Occasional headaches",
        "Dizziness",
        "Fatigue",
        "Stress-related symptoms",
        "Migraine",
        "Light sensitivity",
        "Sleep disturbances",
        "Anxiety symptoms",
        "Mild forgetfulness",
        "Temporary vision spots"
    ]
    
    neuro_exam_abnormal = [
        "Pupil asymmetry noted",
        "Mild hemiparesis observed",
        "Abnormal reflexes",
        "Gait disturbance",
        "Cognitive deficits",
        "Visual field defect",
        "Facial droop present"
    ]
    
    neuro_exam_normal = [
        "No focal deficits",
        "Normal neurological exam",
        "Intact cranial nerves",
        "No motor/sensory deficits",
        "Normal gait and balance"
    ]
    
    tumor_imaging = [
        "MRI: Enhancing mass lesion measuring approximately 2.3 cm in diameter in the left frontal lobe",
        "CT: Hypodense area with surrounding edema in right temporal lobe",
        "MRI: Well-circumscribed extra-axial lesion consistent with meningioma",
        "CT: Irregular hyperdense mass with calcifications in parietal lobe",
        "MRI: Heterogeneously enhancing lesion with mass effect"
    ]
    
    normal_imaging = [
        "MRI: No intracranial mass or abnormal enhancement",
        "CT: Normal brain parenchyma without focal lesions",
        "MRI: Unremarkable study",
        "CT: No acute intracranial findings",
        "MRI: Normal ventricular size and configuration"
    ]
    
    abnormal_labs = [
        "Elevated protein in CSF",
        "Mild leukocytosis",
        "Abnormal hormone levels",
        "Increased inflammatory markers",
        "Electrolyte imbalance"
    ]
    
    normal_labs = [
        "Normal blood work",
        "Routine labs within normal limits",
        "No significant abnormalities",
        "Normal metabolic panel",
        "Unremarkable blood tests"
    ]
    
    medical_conditions = [
        "Hypertension",
        "Diabetes",
        "Migraines",
        "None",
        "Asthma",
        "Depression",
        "High cholesterol",
        "Hypothyroidism"
    ]
    
    for i in range(num_samples):
        has_tumor = np.random.choice([0, 1], p=[0.7, 0.3])
        data['tumor_present'].append(has_tumor)
        
        if has_tumor:
            # Generate tumor case
            data['symptoms'].append(", ".join(np.random.choice(tumor_symptoms, size=np.random.randint(2, 5), replace=False)))
            data['neurological_exam'].append(np.random.choice(neuro_exam_abnormal) + "; " + np.random.choice(neuro_exam_abnormal))
            data['imaging_report'].append(np.random.choice(tumor_imaging))
            data['lab_results'].append(np.random.choice(abnormal_labs))
        else:
            # Generate non-tumor case
            data['symptoms'].append(", ".join(np.random.choice(non_tumor_symptoms, size=np.random.randint(1, 3), replace=False)))
            data['neurological_exam'].append(np.random.choice(neuro_exam_normal))
            data['imaging_report'].append(np.random.choice(normal_imaging))
            data['lab_results'].append(np.random.choice(normal_labs))
        
        data['medical_history'].append(np.random.choice(medical_conditions))
    
    return pd.DataFrame(data)

# Generate and save the dataset
dataset = create_brain_tumor_dataset(500)
dataset.to_csv('brain_tumor_text_dataset.csv', index=False)
print("Dataset with 500 samples created and saved as 'brain_tumor_text_dataset.csv'")