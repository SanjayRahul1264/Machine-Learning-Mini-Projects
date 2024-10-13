import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('best_stock_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the application
st.title("Stock Price Movement Prediction")

# User input fields for stock data
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
close_price = st.number_input("Close Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0)

# Button for prediction
if st.button("Predict"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Close': [close_price],
        'Volume': [volume]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The stock price is predicted to go **up** tomorrow!")
    else:
        st.warning("The stock price is predicted to go **down** tomorrow.")

# Run the application with the command below
# streamlit run your_script_name.py
