import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "scaler_path": BASE_DIR / "scaler1.joblib",
        "required_features": 8,  # Updated to match your scaler
        "features": [
            "gender_male",  # Keep dummy variables in scaling
            "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease",
            "glucose_placeholder", "hba1c_placeholder"  # Padding
        ]
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "scaler_path": BASE_DIR / "scaler2.joblib",
        "required_features": 8,  # Updated to match your scaler
        "features": [
            "gender_male",
            "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease",
            "glucose", "hba1c_placeholder"  # Padding
        ]
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "scaler_path": BASE_DIR / "scaler3.joblib",
        "required_features": 8,  # Updated to match your scaler
        "features": [
            "gender_male",
            "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease",
            "glucose", "hba1c"
        ]
    }
}

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare all features to match scaler expectations"""
    # Convert all inputs
    features = {
        "basic": [
            1 if gender == "Male" else 0,  # gender_male
            float(age), float(bmi),
            {"Never": 0, "Former": 1, "Current": 2}[smoking],
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            0.0, 0.0  # Placeholders
        ],
        "glucose": [
            1 if gender == "Male" else 0,
            float(age), float(bmi),
            {"Never": 0, "Former": 1, "Current": 2}[smoking],
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            float(glucose), 0.0  # Placeholder
        ],
        "full": [
            1 if gender == "Male" else 0,
            float(age), float(bmi),
            {"Never": 0, "Former": 1, "Current": 2}[smoking],
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            float(glucose), float(hba1c)
        ]
    }
    return features

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Load with exact feature validation"""
    resources = {}
    try:
        for name, config in MODEL_CONFIG.items():
            model = load_model(config["model_path"])
            scaler = joblib.load(config["scaler_path"])
            
            # Critical validation
            if scaler.n_features_in_ != config["required_features"]:
                raise ValueError(
                    f"{name}: Scaler has {scaler.n_features_in_} features, "
                    f"config expects {config['required_features']}\n"
                    f"Features: {config['features']}"
                )
            
            resources[name] = {
                "model": model,
                "scaler": scaler,
                "features": config["features"]
            }
        return resources
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    
    # Load resources
    resources = load_resources()
    
    # --- Input Form ---
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 1, 120, 30)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
        with col2:
            st.subheader("Health History")
            smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            
            st.subheader("Advanced Metrics")
            glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 0.0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, 0.1)
        
        submitted = st.form_submit_button("Calculate Risk")
    
    # --- Prediction ---
    if submitted:
        try:
            # Model selection
            model_key = "basic"
            if hba1c > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"
            
            # Get prepared features
            resource = resources[model_key]
            features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )[model_key]
            
            # Scale and predict
            scaled_input = resource["scaler"].transform([features])
            risk = resource["model"].predict(scaled_input)[0][0] * 100
            
            # Display results
            st.success(f"Predicted Risk: {risk:.1f}%")
            st.info(f"""
            **Model Used**: {model_key.upper()}
            **Features**: {resource['features']}
            """)
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
