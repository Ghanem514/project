import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "required_features": 8,  # Matches your model's (None, 8) input shape
        "features": [
            "gender_male",  # Binary (1 for male, 0 for female)
            "age",          # Numeric
            "bmi",          # Numeric
            "smoking_never", "smoking_former", "smoking_current",  # One-hot encoded
            "hypertension", # Binary (1 for yes, 0 for no)
            "heart_disease" # Binary (1 for yes, 0 for no)
        ]
    }
}

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease):
    """Prepare features with exact 8 dimensions model expects"""
    return np.array([[
        1 if gender == "Male" else 0,  # gender_male
        float(age),                   # age
        float(bmi),                   # bmi
        1 if smoking == "Never" else 0,  # smoking_never
        1 if smoking == "Former" else 0, # smoking_former
        1 if smoking == "Current" else 0, # smoking_current
        1 if hypertension == "Yes" else 0, # hypertension
        1 if heart_disease == "Yes" else 0  # heart_disease
    ]], dtype=np.float32)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Diabetes Predictor", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    
    # Load model
    model = load_model(MODEL_CONFIG["basic"]["model_path"])
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 1, 120, 30)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
        with col2:
            st.subheader("Health History")
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        
        submitted = st.form_submit_button("Calculate Risk")
    
    if submitted:
        try:
            # Prepare features
            features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease
            )
            
            # Predict
            risk = model.predict(features)[0][0] * 100
            
            # Display results
            st.success(f"Predicted Diabetes Risk: {risk:.1f}%")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug Info:")
            st.json({
                "input_features": features.tolist(),
                "feature_count": features.shape[1],
                "expected_features": MODEL_CONFIG["basic"]["required_features"]
            })

if __name__ == "__main__":
    main()
