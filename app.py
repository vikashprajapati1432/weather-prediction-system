import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌤️",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the saved pickle model."""
    model_path = 'models/weather_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    st.title("🌤️ Weather Prediction System")
    st.markdown("Enter the weather parameters below to predict if it will rain tomorrow.")

    # Load resources
    model_data = load_model()
    
    if model_data is None:
        st.error("Model file not found! Please run 'train_model.py' first to train and save the model.")
        return

    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']

    # User Inputs in a structured layout
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.number_input("Temperature (°C)", value=25.0, help="Temperature at 3pm")
        humidity = st.number_input("Humidity (%)", value=50.0, min_value=0.0, max_value=100.0, help="Humidity at 3pm")
        
    with col2:
        wind_speed = st.number_input("Wind Speed (km/h)", value=20.0, help="Wind speed at 3pm")
        pressure = st.number_input("Pressure (hPa)", value=1015.0, help="Atmospheric pressure at 3pm")

    rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])
    rain_today_val = 1 if rain_today == "Yes" else 0

    if st.button("Predict Rain Tomorrow"):
        try:
            # Prepare input data
            # Features: ['Temp3pm', 'Humidity3pm', 'WindSpeed3pm', 'Pressure3pm', 'RainToday']
            input_data = np.array([[temp, humidity, wind_speed, pressure, rain_today_val]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]

            # Display Result
            st.markdown("---")
            if prediction[0] == 1:
                st.warning(f"🌧️ **Prediction: It will Rain Tomorrow!**")
                st.info(f"Confidence Level: {probability*100:.2f}%")
            else:
                st.success(f"☀️ **Prediction: No Rain Tomorrow!**")
                st.info(f"Confidence Level: {(1-probability)*100:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Sidebar Information
    st.sidebar.header("About")
    st.sidebar.info("This system uses a Machine Learning model (Random Forest) trained on historical weather data to predict rainfall.")
    st.sidebar.markdown("### Feature Importance")
    st.sidebar.text("1. Humidity\n2. Pressure\n3. Temp\n4. Wind Speed")

if __name__ == "__main__":
    main()
