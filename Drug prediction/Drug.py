import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('best_drug_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit interface
st.title("Drug Prediction App")

# Collect user inputs
age = st.number_input('Enter Age:', min_value=1, max_value=120, value=30)
sex = st.selectbox('Select Sex:', options=['M', 'F'])
bp = st.selectbox('Select Blood Pressure (BP):', options=['LOW', 'HIGH'])
cholesterol = st.selectbox('Select Cholesterol:', options=['LOW', 'HIGH'])
na_to_k = st.number_input('Enter Sodium-to-Potassium ratio (Na_to_K):', min_value=0.0, value=15.0)

# When the user clicks "Predict Drug"
if st.button('Predict Drug'):
    # Encode input values
    sex_encoded = 1 if sex == 'F' else 0
    bp_encoded = 1 if bp == 'HIGH' else 0
    cholesterol_encoded = 1 if cholesterol == 'HIGH' else 0

    # Prepare the input for prediction
    input_data = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])

    # Scale the input data using the saved scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the drug type
    predicted_drug = model.predict(input_data_scaled)
    
    # Display the result
    st.write(f"The predicted drug type is: {predicted_drug[0]}")
