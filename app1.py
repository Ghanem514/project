import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants and Paths ---
BASE_DIR = Path(__file__).parent
MODELS = {
    "basic": BASE_DIR / "model1.h5",
    "glucose": BASE_DIR / "model2.h5",
    "full": BASE_DIR / "model3.h5"
}
SCALERS = {
    "basic": BASE_DIR / "scaler_basic.joblib",
    "glucose": BASE_DIR / "scaler_glucose.joblib",
    "full": BASE_DIR / "scaler_full.joblib"
}

# --- Helper Functions ---
def calculate_bmi(weight_kg, height_cm):
    return weight_kg / ((height_cm / 100) ** 2)

def map_categorical_features(gender, smoking, hypertension, heart_disease):
    gender_encoded = [1, 0] if gender == "Male" else [0, 1]
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    yesno_map = {"No": 0, "Yes": 1}
    return gender_encoded + [smoking_map[smoking]] + [yesno_map[hypertension]] + [yesno_map[heart_disease]]

def select_model(glucose_provided, hba1c_provided):
    if hba1c_provided and glucose_provided:
        return "full", "Model 3 (Glucose + HbA1c)"
    elif glucose_provided:
        return "glucose", "Model 2 (Glucose only)"
    else:
        return "basic", "Model 1 (Basic features)"

@st.cache_resource
def load_models_and_scalers():
    models = {name: load_model(path) for name, path in MODELS.items()}
    scalers = {name: joblib.load(path) for name, path in SCALERS.items()}
    return models, scalers

# --- Main App ---
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("""
    This tool predicts your risk of developing diabetes based on health metrics.  
    For more accurate results, provide glucose and HbA1c levels if available.
    """)

    with st.form("user_inputs"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
            bmi = calculate_bmi(weight, height)
            st.metric("BMI", f"{bmi:.1f}")

        with col2:
            st.subheader("Health History")
            smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            st.subheader("Optional Advanced Metrics")
            glucose = st.number_input("Blood Glucose (mg/dL) - Optional", min_value=0.0, max_value=500.0, value=0.0)
            hba1c = st.number_input("HbA1c (%) - Optional", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

        submitted = st.form_submit_button("Calculate Risk")

    if submitted:
        glucose_provided = glucose > 0
        hba1c_provided = hba1c > 0

        model_key, model_name = select_model(glucose_provided, hba1c_provided)
        models, scalers = load_models_and_scalers()

        base_features = map_categorical_features(gender, smoking, hypertension, heart_disease)
        features = base_features + [age, bmi]
        if glucose_provided:
            features.append(glucose)
        if hba1c_provided:
            features.append(hba1c)

        input_array = np.array([features])
        scaler = scalers[model_key]
        model = models[model_key]

        try:
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0][0] * 100

            st.success("### Prediction Results")
            st.metric("Selected Model", model_name)
            st.metric("Diabetes Risk Probability", f"{prediction:.1f}%")

            st.info("""
            **Risk Interpretation:**
            - < 5%: Low risk  
            - 5–20%: Moderate risk  
            - > 20%: High risk
            """)
        except ValueError as e:
            st.error("There was an issue with the input shape. Please double-check your inputs.")
            st.exception(e)

if __name__ == "__main__":
    main()

