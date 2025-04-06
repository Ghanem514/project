import streamlit as st
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout

# ======================
# MODEL ARCHITECTURES
# ======================

def build_model1():
    model = Sequential([
        InputLayer(input_shape=(7,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_model2():
    model = Sequential([
        InputLayer(input_shape=(8,)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_model3():
    model = Sequential([
        InputLayer(input_shape=(9,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# ======================
# MODEL LOADING
# ======================

@st.cache_resource
def load_models():
    models = {
        'model1': build_model1(),
        'model2': build_model2(),
        'model3': build_model3()
    }
    for name, model in models.items():
        model.load_weights(f'{name}.h5')
    return models

# ======================
# PREPROCESSING
# ======================

@st.cache_resource
def load_preprocessors():
    return {
        'scaler': load('minmax_scaler.joblib'),
        'encoder': load('gender_encoder.joblib')
    }

def preprocess(input_data, model_type):
    preprocessors = load_preprocessors()
    smoking_mapping = {"Never": 1, "Former": 2, "Current": 3, "No Info": 0}
    
    hba1c = input_data['hba1c'] if input_data['hba1c'] is not None else 5.4
    glucose = input_data['glucose'] if input_data['glucose'] is not None else 100
    
    numerical = np.array([
        [input_data['age']],
        [input_data['bmi']],
        [input_data['hypertension']],
        [input_data['heart_disease']],
        [hba1c],
        [glucose]
    ])
    
    numerical_scaled = preprocessors['scaler'].transform(numerical.T)
    gender_encoded = preprocessors['encoder'].transform([[input_data['gender']]]).toarray()
    smoking_encoded = smoking_mapping[input_data['smoking']]
    
    if model_type == "model1":
        return np.concatenate([numerical_scaled[:, :4], gender_encoded, [[smoking_encoded]]], axis=1)
    elif model_type == "model2":
        return np.concatenate([numerical_scaled[:, :5], gender_encoded, [[smoking_encoded]]], axis=1)
    else:
        return np.concatenate([numerical_scaled, gender_encoded, [[smoking_encoded]]], axis=1)

# ======================
# STREAMLIT UI
# ======================

st.title("Diabetes Risk Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age*", min_value=1, max_value=120, value=45)
        bmi = st.number_input("BMI*", min_value=10.0, max_value=50.0, value=25.0)
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=None)
        
    with col2:
        gender = st.selectbox("Gender*", ["Male", "Female"])
        smoking = st.selectbox("Smoking History*", ["Never", "Former", "Current", "No Info"])
        glucose = st.number_input("Blood Glucose", min_value=50, max_value=300, value=None)
    
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    
    model_choice = st.radio("Select Model*", 
                          ["Model 1 (No HbA1c/Glucose)", 
                           "Model 2 (No HbA1c)", 
                           "Model 3 (All Features)"])
    
    # PROPER SUBMIT BUTTON IMPLEMENTATION
    submitted = st.form_submit_button("Calculate Risk")
    st.markdown("*Required fields")

    if submitted:
        # Validate inputs based on model choice
        if model_choice == "Model 3 (All Features)" and (hba1c is None or glucose is None):
            st.error("Please enter both HbA1c and Glucose for Model 3")
        elif model_choice == "Model 2 (No HbA1c)" and glucose is None:
            st.error("Please enter Glucose for Model 2")
        else:
            input_data = {
                'age': age,
                'bmi': bmi,
                'hypertension': int(hypertension),
                'heart_disease': int(heart_disease),
                'hba1c': hba1c,
                'glucose': glucose,
                'gender': gender,
                'smoking': smoking
            }
            
            model_type = {
                "Model 1 (No HbA1c/Glucose)": "model1",
                "Model 2 (No HbA1c)": "model2",
                "Model 3 (All Features)": "model3"
            }[model_choice]
            
            models = load_models()
            
            with st.spinner('Processing...'):
                try:
                    processed_input = preprocess(input_data, model_type)
                    prediction = models[model_type].predict(processed_input)
                    risk_percent = prediction[0][0] * 100
                    
                    st.success(f"Predicted Diabetes Risk: {risk_percent:.1f}%")
                    st.progress(min(int(risk_percent), 100))
                    
                    if risk_percent < 30:
                        st.info("Low risk range")
                    elif 30 <= risk_percent < 70:
                        st.warning("Moderate risk range")
                    else:
                        st.error("High risk range")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# Sidebar with model info
with st.sidebar:
    st.header("Model Guide")
    st.markdown("""
    - **Model 1**: Uses age, BMI, hypertension, heart disease
    - **Model 2**: Adds glucose to Model 1 features
    - **Model 3**: Uses all features including HbA1c
    """)
    st.markdown("Missing values are replaced with dataset averages")