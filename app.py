# app.py
import streamlit as st
import numpy as np
import joblib
import base64

# --- Page config ---
st.set_page_config(page_title="🌾 Crop Recommendation System", page_icon="🌿", layout="centered")

# --- Load model and tools ---
rf = joblib.load("rf_crop_model.joblib")
le = joblib.load("label_encoder.joblib")
scaler = joblib.load("scaler.joblib")

# --- Background & CSS styling ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("assets/background.jpg")  # path to your image

# --- Custom CSS for UI styling ---
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #2E8B57;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stNumberInput > div > input {
        background-color: #ffffffaa;
        color: black;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("assets/logo.png", width=100)
st.title("🌾 Crop Recommendation System")
st.write("### Enter soil and environmental details below:")

# --- Input fields ---
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, step=0.1)

# --- Prediction ---
if st.button("🌱 Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    pred_idx = rf.predict(input_scaled)[0]
    pred_crop = le.inverse_transform([pred_idx])[0]
    st.success(f"🌿 Recommended Crop: **{pred_crop.capitalize()}**")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by <b>Your Name</b> 🌾 | Machine Learning Mini Project</p>", unsafe_allow_html=True)
