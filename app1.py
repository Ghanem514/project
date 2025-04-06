import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "basic": {
        "model_path": BASE_DIR / "model1.h5",
        "scaler_path": BASE_DIR / "scaler1.joblib",
        "required_features": 7,
        "features": [
            "gender", "age", "bmi", 
            "smoking", "hypertension", 
            "heart_disease", "padding"
        ]
    },
    "glucose": {
        "model_path": BASE_DIR / "model2.h5",
        "scaler_path": BASE_DIR / "scaler2.joblib",
        "required_features": 8,
        "features": [
            "gender", "age", "bmi",
            "smoking", "hypertension",
            "heart_disease", "glucose",
            "padding"
        ]
    },
    "full": {
        "model_path": BASE_DIR / "model3.h5",
        "scaler_path": BASE_DIR / "scaler3.joblib",
        "required_features": 9,
        "features": [
            "gender", "age", "bmi",
            "smoking", "hypertension",
            "heart_disease", "glucose",
            "hba1c", "padding"
        ]
    }
}

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare raw features with proper encoding"""
    # Convert all inputs to numerical values
    features = {
        "gender": 1 if gender == "Male" else 0,
        "age": float(age),
        "bmi": float(bmi),
        "smoking": {"Never": 0, "Former": 1, "Current": 2}[smoking],
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "glucose": float(glucose),
        "hba1c": float(hba1c),
        "padding": 0.0  # For dimensional consistency
    }
    
    # Create feature arrays for each model
    return {
        "basic": [features[k] for k in MODEL_CONFIG["basic"]["features"]],
        "glucose": [features[k] for k in MODEL_CONFIG["glucose"]["features"]],
        "full": [features[k] for k in MODEL_CONFIG["full"]["features"]]
    }

# --- Model Loading ---
@st.cache_resource
def load_resources():
    """Load models and scalers with validation"""
    resources = {}
    try:
        for name, config in MODEL_CONFIG.items():
            # Load model and scaler
            model = load_model(config["model_path"])
            scaler = joblib.load(config["scaler_path"])
            
            # Verify dimensions
            if scaler.n_features_in_ != config["required_features"]:
                raise ValueError(
                    f"{name}: Scaler expects {scaler.n_features_in_} features, "
                    f"but config specifies {config['required_features']}"
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
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, step=0.1)
        
        submitted = st.form_submit_button("Calculate Risk")
    
    # --- Prediction ---
    if submitted:
        try:
            # Determine which models to use
            active_models = ["basic"]
            if glucose > 0:
                active_models.append("glucose")
            if hba1c > 0:
                active_models.append("full")
            
            # Prepare features
            raw_features = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )
            
            # Make predictions
            results = []
            for model_key in active_models:
                resource = resources[model_key]
                
                # Scale features
                features = np.array([raw_features[model_key]], dtype=np.float32)
                scaled_features = resource["scaler"].transform(features)
                
                # Predict
                risk_percentage = resource["model"].predict(scaled_features)[0][0] * 100
                results.append({
                    "model": model_key,
                    "risk": risk_percentage,
                    "features": resource["features"]
                })
            
            # Display results
            st.success("### Prediction Results")
            
            for result in results:
                st.metric(
                    label=f"{result['model'].upper()} Model",
                    value=f"{result['risk']:.1f}%",
                    help=f"Features: {', '.join(result['features'])}"
                )
            
            # Interpretation guide
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
