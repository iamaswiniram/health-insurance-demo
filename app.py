import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("""
    Enter the patient's details below to get an estimated medical insurance charge.
""")

# Load the trained model, scaler, and encoder
try:
    best_model = joblib.load('best_insurance_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    numerical_features = joblib.load('numerical_features.pkl')
    categorical_features = joblib.load('categorical_features.pkl')
    encoded_feature_names = joblib.load('encoded_feature_names.pkl')

except FileNotFoundError:
    st.error("Error: Model or preprocessors not found. Please run 'insurance_prediction_no_pipeline.py' first to train and save them.")
    st.stop()

# Input fields for user
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.slider("BMI", 15.0, 55.0, 25.0, step=0.1)

with col2:
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction button
if st.button("Predict Insurance Charges"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # --- Manual Preprocessing for the input data ---
    # 1. Scale numerical features
    input_numerical_scaled = scaler.transform(input_data[numerical_features])
    input_numerical_df = pd.DataFrame(input_numerical_scaled, columns=numerical_features, index=input_data.index)

    # 2. One-hot encode categorical features
    input_categorical_encoded = encoder.transform(input_data[categorical_features])
    input_categorical_df = pd.DataFrame(input_categorical_encoded, columns=encoded_feature_names, index=input_data.index)

    # 3. Concatenate all processed features
    processed_input = pd.concat([input_numerical_df, input_categorical_df], axis=1)

    # Make prediction using the loaded model
    predicted_log_charges = best_model.predict(processed_input)

    # Inverse transform the prediction to get actual charges
    predicted_charges = np.expm1(predicted_log_charges)[0]

    st.subheader("Predicted Medical Insurance Charges:")
    st.success(f"**${predicted_charges:,.2f}**")
    st.markdown("---")
    st.info("Disclaimer: This is an estimated cost based on the provided data and model. Actual costs may vary.")

st.markdown("---")
st.markdown("Developed by Ramasamy_A_Batch_11 for a Mini Project-Supervised ML")






