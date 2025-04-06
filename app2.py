import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# --- Constants ---
BASE_DIR = Path(__file__).parent
MODELS = {
    "basic": BASE_DIR / "model1.h5",      # 7 features
    "glucose": BASE_DIR / "model2.h5",    # 8 features
    "full": BASE_DIR / "model3.h5"        # 9 features
}
SCALERS = {
    "basic": BASE_DIR / "scaler_basic.joblib",    # 7 features
    "glucose": BASE_DIR / "scaler_glucose.joblib", # 8 features
    "full": BASE_DIR / "scaler_full.joblib"       # 9 features
}

# --- Model Loading ---
@st.cache_resource
def load_resources():
    """Load models and scalers with validation"""
    try:
        models = {}
        scalers = {}
        
        # Load all models and their corresponding scalers
        for name in MODELS:
            models[name] = load_model(MODELS[name])
            scalers[name] = joblib.load(SCALERS[name])
            
            # Verify feature counts match
            expected_features = {
                "basic": 7,
                "glucose": 8,
                "full": 9
            }[name]
            
            if scalers[name].n_features_in_ != expected_features:
                raise ValueError(
                    f"Model {name} expects {expected_features} features, "
                    f"but scaler has {scalers[name].n_features_in_}"
                )
                
        return scalers, models
        
    except Exception as e:
        st.error(f"❌ Failed to load resources: {str(e)}")
        st.stop()

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Create feature array with consistent order"""
    # Convert categorical features
    gender_encoded = [1, 0] if gender == "Male" else [0, 1]
    smoking_encoded = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    
    # Base features for basic model
    features = [
        *gender_encoded,
        float(age),
        float(bmi),
        smoking_encoded,
        int(hypertension == "Yes"),
        int(heart_disease == "Yes")
    ]
    
    # Add glucose for intermediate model
    if glucose > 0:
        features.append(float(glucose))
        
    # Add HbA1c for full model
    if hba1c > 0:
        features.append(float(hba1c))
        
    return np.array([features])

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    
    # Load resources
    scalers, models = load_resources()
    
    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
        with col2:
            st.subheader("Health History")
            smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            
            st.subheader("Advanced Metrics (Optional)")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, value=0.0)
            hba1c = st.number_input("HbA1c (%)", min_value=0.0, value=0.0, step=0.1)
        
        submitted = st.form_submit_button("Calculate Risk")
    
    # --- Prediction ---
    if submitted:
        try:
            # Determine which model to use
            model_key = "basic"
            if hba1c > 0 and glucose > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"
            
            # Prepare features
            input_array = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )
            
            # Get the correct scaler and model
            scaler = scalers[model_key]
            model = models[model_key]
            
            # Validate feature count
            expected_features = {
                "basic": 7,
                "glucose": 8,
                "full": 9
            }[model_key]
            
            if input_array.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {input_array.shape[1]}"
                )
            
            # Scale and predict
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0][0] * 100
            
            # Display results
            st.success("### Prediction Results")
            cols = st.columns(3)
            cols[0].metric("Selected Model", model_key.capitalize())
            cols[1].metric("Features Used", expected_features)
            cols[2].metric("Diabetes Risk", f"{prediction:.1f}%")
            
            # Interpretation guide
            st.info("""
            **Risk Interpretation:**
            - < 5%: Low risk
            - 5-20%: Moderate risk  
            - > 20%: High risk
            """)
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
            st.write("Debug Info:")
            st.json({
                "input_features": input_array.tolist()[0],
                "expected_features": expected_features,
                "actual_features": input_array.shape[1],
                "model_used": model_key
            })

if __name__ == "__main__":
    main()
