# crop_recommendation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load dataset
df = pd.read_csv("Crop_recommendation.csv")  # file in same folder
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# 2️⃣ Preprocessing
X = df.drop(columns=['label'])
y = df['label']

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4️⃣ Model training
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("✅ Model training complete!")

# 5️⃣ Evaluation
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 6️⃣ Save the model and tools
joblib.dump(rf, "rf_crop_model.joblib")
joblib.dump(le, "label_encoder.joblib")
joblib.dump(scaler, "scaler.joblib")
print("\n💾 Model, label encoder, and scaler saved successfully!")

# 7️⃣ Visualize feature importance
plt.figure(figsize=(6,4))
sns.barplot(x=df.columns[:-1], y=rf.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in Crop Prediction")
plt.tight_layout()
plt.show()
