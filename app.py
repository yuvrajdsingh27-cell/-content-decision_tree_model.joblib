import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", page_icon="🚚", layout="centered")

# Load the pre-trained Decision Tree model
@st.cache_resource
def load_model():
    return joblib.load('decision_tree_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}. Please ensure 'decision_tree_model.joblib' is in the same directory.")
    st.stop()

# Title and description
st.title("🚚 Delivery Delay Prediction App")
st.write("Enter the shipment details below to predict whether a delivery is likely to be **On Time** or **Delayed**.")

# Create input form
with st.form("prediction_form"):
    st.subheader("Shipment & Route Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, value=25.0, step=1.0)
        traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition Severity (1-5)", min_value=1, max_value=5, value=2)
        delivery_slot = st.slider("Delivery Slot (Shift/Time block)", min_value=1, max_value=4, value=2)
        driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=5)
        num_stops = st.number_input("Number of Stops on Route", min_value=0, max_value=20, value=3)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=4)
        road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
        package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, value=15.0, step=0.5)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=12.5, step=0.5)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, value=45)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Status")

if submit_button:
    # Map inputs into a dictionary matching the training feature names
    input_data = {
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }
    
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Generate Prediction
    prediction = model.predict(input_df)[0]
    
    # Try to predict probability if decision tree supports it
    try:
        prediction_proba = model.predict_proba(input_df)[0]
        confidence = prediction_proba[prediction] * 100
    except:
        confidence = None

    st.markdown("--- ")
    if prediction == 1:
        st.error("### ⚠️ Prediction: Delayed")
        if confidence is not None:
            st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.success("### ✅ Prediction: On Time")
        if confidence is not None:
            st.write(f"Confidence: {confidence:.2f}%")
