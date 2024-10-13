# Import necessary libraries for Streamlit
import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app title
st.title("Spam Detection App")

# Input from the user
message = st.text_area("Enter the message to classify:")

# Button to predict
if st.button("Predict"):
    # Transform the input message using the vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(message_tfidf)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The message is **Spam**")
    else:
        st.success("The message is **Ham**")

# Optionally, provide some example messages
st.subheader("Example Messages")
st.write("1. Free entry in 2 a wkly comp to win FA Cup final... (Spam)")
st.write("2. Go until jurong point, crazy.. Available only ... (Ham)")
