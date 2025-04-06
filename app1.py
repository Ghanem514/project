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
    "basic": BASE_DIR / "scaler1.joblib",    # 7 features
    "glucose": BASE_DIR / "scaler2.joblib",  # 8 features
    "full": BASE_DIR / "scaler3.joblib"      # 9 features
}

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare features with proper encoding"""
    # 1. Gender encoding (Male=1, Female=0)
    gender_encoded = 1 if gender == "Male" else 0
    
    # 2. Smoking history (Never=0, Former=1, Current=2)
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    smoking_encoded = smoking_map[smoking]
    
    # 3. Yes/No features (Yes=1, No=0)
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    
    # Base features (7)
    features = [
        gender_encoded,
        float(age),
        float(bmi),
        smoking_encoded,
        hypertension_encoded,
        heart_disease_encoded
    ]
    
    # Add glucose if provided (8)
    if glucose > 0:
        features.append(float(glucose))
        
    # Add HbA1c if provided (9)
    if hba1c > 0:
        features.append(float(hba1c))
        
    return np.array([features])

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Load all models and their matching scalers"""
    try:
        resources = {}
        for name in MODELS:
            # Load model and its specific scaler
            model = load_model(MODELS[name])
            scaler = joblib.load(SCALERS[name])
            
            # Verify feature counts match
            expected_features = {
                "basic": 7,
                "glucose": 8,
                "full": 9
            }[name]
            
            if scaler.n_features_in_ != expected_features:
                raise ValueError(
                    f"Model {name} expects {expected_features} features, "
                    f"but scaler has {scaler.n_features_in_}"
                )
                
            resources[name] = {"model": model, "scaler": scaler}
            
        return resources
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

# --- Streamlit UI ---
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
            smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            
            st.subheader("Advanced Metrics")
            glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 0.0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, 0.1)
        
        submitted = st.form_submit_button("Calculate Risk")
    
    # --- Prediction ---
    if submitted:
        try:
            # Determine which model to use
            model_key = "basic"
            if hba1c > 0:
                model_key = "full"
            elif glucose > 0:
                model_key = "glucose"
            
            # Get the correct resources
            model = resources[model_key]["model"]
            scaler = resources[model_key]["scaler"]
            
            # Prepare features
            input_array = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )
            
            # Validate feature count
            if input_array.shape[1] != scaler.n_features_in_:
                raise ValueError(
                    f"Expected {scaler.n_features_in_} features, got {input_array.shape[1]}"
                )
            
            # Scale and predict
            scaled_input = scaler.transform(input_array)
            risk = model.predict(scaled_input)[0][0] * 100
            
            # Display results
            st.success(f"Predicted Risk: {risk:.1f}%")
            st.info(f"""
            **Model Used**: {model_key.upper()} ({scaler.n_features_in_} features)
            **Risk Interpretation**:
            - < 5%: Low risk
            - 5-20%: Moderate risk  
            - > 20%: High risk
            """)
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
            st.json({
                "error_details": {
                    "expected_features": resources[model_key]["scaler"].n_features_in_,
                    "received_features": input_array.shape[1],
                    "input_values": input_array.tolist()[0]
                }
            })

if __name__ == "__main__":
    main()
