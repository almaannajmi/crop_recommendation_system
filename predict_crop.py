# predict_crop.py
import joblib
import numpy as np

# Load saved model and tools
rf = joblib.load("rf_crop_model.joblib")
le = joblib.load("label_encoder.joblib")
scaler = joblib.load("scaler.joblib")

# Example input: [N, P, K, temperature, humidity, ph, rainfall]
sample = np.array([[90, 42, 43, 20.5, 80, 6.5, 200]])
sample_scaled = scaler.transform(sample)

# Predict crop
pred_idx = rf.predict(sample_scaled)[0]
pred_crop = le.inverse_transform([pred_idx])[0]

print("🌾 Recommended crop:", pred_crop)
