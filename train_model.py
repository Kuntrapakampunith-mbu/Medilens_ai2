import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("🚀 Starting MediLens AI Model Training Pipeline...")
print("📥 Downloading dataset...")

# Load UCI Cleveland Heart Disease dataset (or your local dataset path)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv(url, names=column_names)

print("🧹 Cleaning dataset (handling missing values)...")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)

# Convert target to binary (0 = No disease, 1 = Disease present)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and features
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/heart_disease_model.pkl')
joblib.dump(list(X.columns), 'model/model_features.pkl')

print("💾 Model and features saved successfully to 'model/' directory!")