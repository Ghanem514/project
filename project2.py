import streamlit as st
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout

# [Previous code for model architectures, loading, and preprocessing remains EXACTLY the same...]

# ======================
# STREAMLIT UI - CORRECTED FORM IMPLEMENTATION
# ======================

st.title("Diabetes Risk Prediction")

# Initialize form
form = st.form(key='prediction_form')

with form:
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
    
    # The critical fix - properly defined submit button
    submit_button = st.form_submit_button(label='Calculate Risk')
    st.markdown("*Required fields")

# Handle form submission OUTSIDE the form context
if submit_button:
    # Validate inputs
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

# [Rest of your code (sidebar, etc.) remains the same...]