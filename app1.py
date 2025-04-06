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
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease"
        ]
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "scaler_path": BASE_DIR / "scaler2.joblib",
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose"
        ]
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "scaler_path": BASE_DIR / "scaler3.joblib",
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose", "hba1c"
        ]
    }
}

# --- Feature Preparation ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare features for all model types"""
    gender_encoded = 1 if gender == "Male" else 0
    smoking_encoded = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0

    features = {
        "basic": [
            gender_encoded, float(age), float(bmi), smoking_encoded,
            hypertension_encoded, heart_disease_encoded
        ],
        "glucose": [
            gender_encoded, float(age), float(bmi), smoking_encoded,
            hypertension_encoded, heart_disease_encoded,
            float(glucose)
        ],
        "full": [
            gender_encoded, float(age), float(bmi), smoking_encoded,
            hypertension_encoded, heart_disease_encoded,
            float(glucose), float(hba1c)
        ]
    }
    return features

# --- Model Loading ---
@st.cache_resource
def load_resources():
    """Load models and scalers with validation"""
    resources = {}
    try:
        for name, config in MODEL_CONFIG.items():
            model = load_model(config["model_path"])
            scaler = joblib.load(config["scaler_path"])

            # Validate feature count
            if scaler.n_features_in_ != len(config["features"]):
                raise ValueError(
                    f"{name}: Scaler expects {scaler.n_features_in_} features, "
                    f"but config specifies {len(config['features'])}"
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
            bmi = weight / ((height / 100) ** 2)
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
            # Decide which model to use
            model_key = "basic"
            if hba1c > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"

            resource = resources[model_key]
            input_features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease, glucose, hba1c
            )[model_key]

            # Scale features
            scaled_features = resource["scaler"].transform([input_features])

            # Predict
            prediction = resource["model"].predict(scaled_features)[0][0]
            risk_percent = prediction * 100

            # Display results
            st.success(f"Predicted Risk: {risk_percent:.1f}%")
            st.info(f"""
            **Model Used**: {model_key.upper()}  
            **Features Used**: {', '.join(resource['features'])}
            """)
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()

