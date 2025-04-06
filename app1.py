import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "required_features": 8,
        "features": [
            "gender_male", "age", "bmi",
            "smoking_never", "smoking_former", "smoking_current",
            "hypertension", "heart_disease"
        ],
        "description": "Basic model (no glucose/HbA1c)"
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "required_features": 9,
        "features": [
            "gender_male", "age", "bmi",
            "smoking_never", "smoking_former", "smoking_current",
            "hypertension", "heart_disease", "glucose"
        ],
        "description": "Includes blood glucose"
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "required_features": 10,
        "features": [
            "gender_male", "age", "bmi",
            "smoking_never", "smoking_former", "smoking_current",
            "hypertension", "heart_disease", "glucose", "hba1c"
        ],
        "description": "Includes glucose + HbA1c"
    }
}

# --- Feature Preparation ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare features for all models with proper encoding"""
    base_features = [
        1 if gender == "Male" else 0,  # gender_male
        float(age),                   # age
        float(bmi),                   # bmi
        1 if smoking == "Never" else 0,   # smoking_never
        1 if smoking == "Former" else 0,  # smoking_former
        1 if smoking == "Current" else 0, # smoking_current
        1 if hypertension == "Yes" else 0,  # hypertension
        1 if heart_disease == "Yes" else 0   # heart_disease
    ]
    
    return {
        "basic": np.array([base_features], dtype=np.float32),
        "glucose": np.array([base_features + [float(glucose)]], dtype=np.float32),
        "full": np.array([base_features + [float(glucose), float(hba1c)]], dtype=np.float32)
    }

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load all models with validation"""
    models = {}
    for name, config in MODEL_CONFIG.items():
        try:
            model = load_model(config["model_path"])
            if model.input_shape[1] != config["required_features"]:
                st.error(f"Model {name} expects {model.input_shape[1]} features, but config specifies {config['required_features']}")
                st.stop()
            models[name] = model
        except Exception as e:
            st.error(f"Failed to load {name}: {str(e)}")
            st.stop()
    return models

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    
    # Load models
    models = load_models()
    
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
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"] )
            
            st.subheader("Advanced Metrics")
            glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 0.0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, step=0.1)
        
        submitted = st.form_submit_button("Calculate Risk")
    
    # --- Prediction ---
    if submitted:
        try:
            # Prepare features
            features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )
            
            # Log the features to check the values passed to the model
            st.write("Prepared features for the model:")
            st.write(features)
            st.write(f"Shape of the input array: {features['basic'].shape}")
            
            # Determine which models to use
            active_models = ["basic"]
            if glucose > 0:
                active_models.append("glucose")
            if hba1c > 0:
                active_models.append("full")
            
            # Display results
            st.success("### Prediction Results")
            
            for model_key in active_models:
                prediction = models[model_key].predict(features[model_key])[0][0] * 100
                st.metric(
                    label=f"{MODEL_CONFIG[model_key]['description']}",
                    value=f"{prediction:.1f}%",
                    help=f"Features used: {', '.join(MODEL_CONFIG[model_key]['features'])}"
                )
            
            # Risk interpretation guide
            st.info("""
            **Risk Interpretation:**
            - < 5%: Low risk
            - 5-20%: Moderate risk  
            - > 20%: High risk
            """)
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()

