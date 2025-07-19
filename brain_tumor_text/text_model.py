import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('brain_tumor_text_dataset2.csv')

# Feature engineering: Combine symptoms and neurological exam text for text-based features
data['combined_text'] = data['symptoms'] + " " + data['neurological_exam']

# Modify data to reduce accuracy: 
# Introduce noise by randomly changing some symptom labels (to confuse the model)
for i in range(len(data)):
    if np.random.rand() > 0.95:  # Add noise for about 5% of data
        data.loc[i, 'symptoms'] = "Random noise symptoms"
        data.loc[i, 'neurological_exam'] = "Random noise neurological exam"
        
# Remove or modify some important features (e.g., remove medical history)
data = data.drop(columns=['medical_history'])

# Prepare the features (X) and target (y)
X = data[['age', 'gender', 'family_history', 'combined_text']]
y = data['tumor_present']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'family_history']),
        ('text', TfidfVectorizer(max_features=1000), 'combined_text')
    ])

# Build the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'brain_tumor_model_text.pkl')
print("\nModel saved as 'brain_tumor_model_text.pkl'")
