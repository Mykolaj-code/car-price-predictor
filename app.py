# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the model and columns
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Car Price Predictor 🚗")
st.write("Enter the car's features for a prediction.")

# --- Create input fields based on REAL data ---
st.sidebar.header("Car Features")

# Numerical fields (using sliders for convenience)
horsepower = st.sidebar.slider("Horsepower (hp)", 50, 250, 100)
city_mpg = st.sidebar.slider("Fuel Economy (city, mpg)", 15, 50, 25)
highway_mpg = st.sidebar.slider("Fuel Economy (highway, mpg)", 15, 50, 30)
wheel_base = st.sidebar.slider("Wheel Base (inches)", 85.0, 125.0, 95.0)


# Categorical fields (with options from your dataset)
drive_wheels = st.sidebar.selectbox("Drive Wheels", ['rwd', 'fwd', '4wd'])
fuel_type = st.sidebar.selectbox("Fuel Type", ['gas', 'diesel'])
body_style = st.sidebar.selectbox("Body Style", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])


# Create a button for prediction
if st.sidebar.button("Predict Price"):
    # Create a DataFrame from the input
    # IMPORTANT: The dictionary keys EXACTLY match the column names
    input_data = pd.DataFrame({
        'wheel-base': [wheel_base],
        'horsepower': [horsepower],
        'city-mpg': [city_mpg],
        'highway-mpg': [highway_mpg],
        'drive-wheels': [drive_wheels],
        'fuel-type': [fuel_type],
        'body-style': [body_style]
    })

    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Align columns
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(final_input)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")