import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease"
        ]
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose"
        ]
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "features": [
            "gender_encoded", "age", "bmi", "smoking_encoded",
            "hypertension", "heart_disease", "glucose", "hba1c"
        ]
    }
}

# --- Manual Mean/Std (replace these with actual training data stats!) ---
FEATURE_STATS = {
    "age": (50, 15),
    "bmi": (25, 5),
    "smoking_encoded": (1, 0.8),
    "hypertension": (0.2, 0.4),
    "heart_disease": (0.1, 0.3),
    "gender_encoded": (0.5, 0.5),
    "glucose": (100, 30),
    "hba1c": (5.5, 1.5)
}

def standardize(value, mean, std):
    return (value - mean) / std if std != 0 else value

# --- Feature Preparation ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    gender_encoded = 1 if gender == "Male" else 0
    smoking_encoded = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0

    raw_features = {
        "gender_encoded": gender_encoded,
        "age": float(age),
        "bmi": float(bmi),
        "smoking_encoded": smoking_encoded,
        "hypertension": hypertension_encoded,
        "heart_disease": heart_disease_encoded,
        "glucose": float(glucose),
        "hba1c": float(hba1c)
    }

    # Create all 3 feature sets, standardized
    features = {
        "basic": [],
        "glucose": [],
        "full": []
    }
    for key in MODEL_CONFIG["basic"]["features"]:
        features["basic"].append(standardize(raw_features[key], *FEATURE_STATS[key]))
    for key in MODEL_CONFIG["glucose"]["features"]:
        features["glucose"].append(standardize(raw_features[key], *FEATURE_STATS[key]))
    for key in MODEL_CONFIG["full"]["features"]:
        features["full"].append(standardize(raw_features[key], *FEATURE_STATS[key]))

    return features

# --- Load Models ---
@st.cache_resource
def load_models():
    models = {}
    for name, config in MODEL_CONFIG.items():
        models[name] = load_model(config["model_path"])
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
            # Choose model
            model_key = "basic"
            if hba1c > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"

            model = models[model_key]
            features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )[model_key]

            prediction = model.predict([features])[0][0] * 100
            st.success(f"Predicted Risk: {prediction:.1f}%")
            st.info(f"Model: {model_key.upper()}, Features used: {MODEL_CONFIG[model_key]['features']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
