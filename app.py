# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and column names
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Used Car Price Predictor 🚗")
st.write("Enter the car's features to get a price prediction.")

# Create input fields in the sidebar
st.sidebar.header("Car Features")
year = st.sidebar.number_input("Year", 1990, 2024, 2017)
present_price = st.sidebar.number_input("Current Showroom Price (in $1000s)", 1.0, 50.0, 5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", 1, 200000, 50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Create a button to make a prediction
if st.sidebar.button("Predict Price"):
    # Create a DataFrame from the user's input
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    # One-Hot Encode the input data
    input_encoded = pd.get_dummies(input_data)

    # Align the columns of the input data with the model's columns
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(final_input)

    # Display the result
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")