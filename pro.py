import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load scaler and models
scaler = joblib.load("minmax_scaler.joblib")

model1 = load_model("model1.h5")  # without glucose & hba1c
model2 = load_model("model2.h5")  # with glucose
model3 = load_model("model3.h5")  # with glucose + hba1c

# Helper functions
def map_smoking(smoking):
    return {"never": 0, "former": 1, "current": 2}.get(smoking.lower(), 0)

def map_yesno(value):
    return 1 if value.lower() == "yes" else 0

def encode_gender(gender):
    return [1, 0] if gender == "Male" else [0, 1]

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0)

# Calculate BMI
bmi = weight / ((height / 100) ** 2)
st.markdown(f"**Calculated BMI:** `{bmi:.2f}`")

smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

# Optional fields
glucose = st.number_input("Blood Glucose Level (optional)", min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
hba1c = st.number_input("HbA1c Level (optional)", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")

# Predict button
if st.button("Predict Diabetes Risk"):
    gender_encoded = encode_gender(gender)
    smoking_val = map_smoking(smoking)
    hypertension_val = map_yesno(hypertension)
    heart_disease_val = map_yesno(heart_disease)

    # Decide which model to use
    use_glucose = glucose > 0
    use_hba1c = hba1c > 0

    if use_glucose and use_hba1c:
        model = model3
        input_features = gender_encoded + [age, bmi, smoking_val, hypertension_val, heart_disease_val, glucose, hba1c]
    elif use_glucose:
        model = model2
        input_features = gender_encoded + [age, bmi, smoking_val, hypertension_val, heart_disease_val, glucose]
    else:
        model = model1
        input_features = gender_encoded + [age, bmi, smoking_val, hypertension_val, heart_disease_val]

    # Scale and predict
    input_array = np.array([input_features])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0][0] * 100

    st.success(f"🎯 Predicted Diabetes Risk: **{prediction:.2f}%**")
