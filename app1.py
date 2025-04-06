import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "expected_features": 6,  # Updated to match model1.h5
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease"
        ]
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "expected_features": 7,  # Updated to match model2.h5
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose"
        ]
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "expected_features": 8,  # Updated to match model3.h5
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose", "hba1c"
        ]
    }
}

# --- Feature Preparation ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare features with proper shape for each model"""
    # Convert all inputs
    features = {
        "gender_encoded": 1 if gender == "Male" else 0,
        "age": float(age),
        "bmi": float(bmi),
        "smoking_encoded": {"Never": 0, "Former": 1, "Current": 2}[smoking],
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "glucose": float(glucose),
        "hba1c": float(hba1c)
    }
    
    # Create feature arrays for each model
    return {
        "basic": np.array([
            features["gender_encoded"],
            features["age"],
            features["bmi"],
            features["smoking_encoded"],
            features["hypertension"],
            features["heart_disease"]
        ]).reshape(1, -1),  # Shape (1, 6)
        
        "glucose": np.array([
            features["gender_encoded"],
            features["age"],
            features["bmi"],
            features["smoking_encoded"],
            features["hypertension"],
            features["heart_disease"],
            features["glucose"]
        ]).reshape(1, -1),  # Shape (1, 7)
        
        "full": np.array([
            features["gender_encoded"],
            features["age"],
            features["bmi"],
            features["smoking_encoded"],
            features["hypertension"],
            features["heart_disease"],
            features["glucose"],
            features["hba1c"]
        ]).reshape(1, -1)  # Shape (1, 8)
    }

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load models with validation"""
    models = {}
    for name, config in MODEL_CONFIG.items():
        model = load_model(config["model_path"])
        # Verify model input shape
        if model.input_shape[1] != config["expected_features"]:
            raise ValueError(
                f"Model {name} expects {model.input_shape[1]} features, "
                f"but config specifies {config['expected_features']}"
            )
        models[name] = model
    return models

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")

    models = load_models()

    with st.form("user_input"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Personal Info")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 1, 120, 30)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
            bmi = weight / ((height / 100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
        with col2:
            st.subheader("Medical Info")
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 0.0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, step=0.1)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        try:
            # Determine which model to use
            model_key = "basic"
            if hba1c > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"

            # Get prepared features
            features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )[model_key]

            # Predict
            prediction = model.predict(np.array([features]))[0][0] * 100
            st.success(f"Predicted Risk: {prediction:.1f}%")
            st.info(f"Model: {model_key.upper()}, Features used: {MODEL_CONFIG[model_key]['features']}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Debug Info:")
            st.json({
                "model_key": model_key,
                "features_shape": features.shape,
                "expected_shape": models[model_key].input_shape
            })

if __name__ == "__main__":
    main()
