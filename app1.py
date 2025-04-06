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

# --- Feature Engineering ---
def prepare_features(gender, age, bmi, smoking, hypertension, heart_disease, glucose=0, hba1c=0):
    """Prepare features EXACTLY as done during training"""
    # 1. Gender (dummy encoded: Female=0, Male=1)
    gender_male = 1 if gender == "Male" else 0
    
    # 2. Smoking (ordinal encoded)
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    smoking_encoded = smoking_map[smoking]
    
    # 3. Yes/No features (binary)
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    
    # Base features for Model 1 (7 features)
    features = [
        gender_male,        # 1
        float(age),         # 2
        float(bmi),         # 3
        smoking_encoded,    # 4
        hypertension_encoded, # 5
        heart_disease_encoded # 6
        # (7th feature is added below)
    ]
    
    # Add glucose for Model 2 (8 features)
    if glucose > 0:
        features.append(float(glucose))  # 7
        
    # Add HbA1c for Model 3 (9 features)
    if hba1c > 0:
        features.append(float(hba1c))    # 8
        
    # Pad with zeros if needed (matches training)
    while len(features) < 9:
        features.append(0.0)
        
    return np.array([features[:7], features[:8], features[:9]])  # Return all variants

# --- Model Loading ---
@st.cache_resource
def load_resources():
    """Load models with validation"""
    try:
        models = {name: load_model(path) for name, path in MODELS.items()}
        # Single scaler trained on MAX features (9)
        scaler = joblib.load(BASE_DIR / "scaler.joblib") 
        return scaler, models
    except Exception as e:
        st.error(f"❌ Failed to load: {str(e)}")
        st.stop()

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("🩺 Diabetes Risk Prediction")
    
    # Load resources
    scaler, models = load_resources()
    
    # --- Input Form ---
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            gender = st.selectbox("Gender", ["Female", "Male"])  # Female first!
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
            # Prepare ALL feature variants
            basic_feat, glucose_feat, full_feat = prepare_features(
                gender, age, bmi, smoking,
                hypertension, heart_disease,
                glucose, hba1c
            )
            
            # Select correct features
            if hba1c > 0:
                features = full_feat
                model = models["full"]
            elif glucose > 0:
                features = glucose_feat
                model = models["glucose"]
            else:
                features = basic_feat
                model = models["basic"]
            
            # Scale and predict
            scaled_input = scaler.transform(features.reshape(1, -1))
            risk = model.predict(scaled_input)[0][0] * 100
            
            # Display results
            st.success(f"Predicted Risk: {risk:.1f}%")
            st.info(f"""
            **Model Used**: {"Full" if hba1c > 0 else "Glucose" if glucose > 0 else "Basic"}
            **Features**: {features.shape[1]} 
            """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug Info:")
            st.json({
                "input_features": features.tolist(),
                "scaler_features": scaler.n_features_in_
            })

if __name__ == "__main__":
    main()
