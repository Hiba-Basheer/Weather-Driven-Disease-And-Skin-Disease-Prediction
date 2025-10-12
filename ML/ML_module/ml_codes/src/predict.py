import pandas as pd
import numpy as np
import logging
from utils.preprocess import preprocess_data
from utils.weather_api import get_weather_data

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_symptoms(user_symptoms: list, valid_columns: list) -> list:
    """
    Normalize user symptoms to match training feature format.
    """
    normalized = []
    for symptom in user_symptoms:
        formatted = symptom.lower().strip().replace(" ", "_")
        if formatted in valid_columns:
            normalized.append(formatted)
    logger.info(f"Normalized symptoms: {normalized}")
    return normalized


def predict_disease(user_input: dict, model_path, le_path, symptom_columns, show_top_n=False) -> str:
    """
    Predict disease based on user symptoms and weather data, with confidence handling.

    Args:
        user_input (dict): Dictionary with keys - age, gender, symptoms, and city.
        model_path (str): Path to the trained model file.
        le_path (str): Path to the label encoder file.
        symptom_columns (list): List of known symptom feature columns.
        show_top_n (bool): If True, return top 3 most probable predictions.

    Returns:
        str: Prediction result, including confidence.
    """
    logger.info("Starting disease prediction...")
    
    # Ensure helper functions are available
    try:
        from symptom_mapper import map_symptoms
        from load_model import load_trained_model
    except ImportError:
        logger.error("Required module (symptom_mapper or load_model) not found. Cannot proceed.")
        return "Internal Error: Required prediction helper modules could not be loaded."


    user_input['symptoms'] = normalize_symptoms(user_input['symptoms'], symptom_columns)
    model, label_encoder = load_trained_model(model_path, le_path)
    logger.info("Model and label encoder loaded successfully.")

    
    temperature = None
    humidity = None
    wind_speed = None
    
    # Fetch weather data
    if 'city' in user_input and user_input['city']:
        weather = get_weather_data(user_input['city'])
        logger.info(f"Fetched weather data for city: {user_input['city']}")
        
        temperature = weather['temperature']
        humidity = weather['humidity'] # 0-100%
        
        # OWM returns m/s; convert to km/h
        wind_speed_ms = weather['wind_speed'] 
        wind_speed = wind_speed_ms * 3.6 
        logger.info(f"Converted wind speed: {wind_speed_ms:.2f} m/s -> {wind_speed:.2f} km/h")

    elif user_input.get('temperature') is not None and user_input.get('humidity') is not None and user_input.get('wind_speed') is not None:
        temperature = user_input.get('temperature')
        humidity = user_input.get('humidity')
        wind_speed = user_input.get('wind_speed')
        logger.info("Using raw weather data from user input (assuming km/h for wind speed).")
    else:
        logger.error("Insufficient weather data (city or raw values) provided.")
        return "Error: Please provide a city name or raw weather values (temperature, humidity, wind speed)."

    # Prepare DataFrame
    input_data = {
        'Age': [user_input['age']],
        'Gender': [1 if user_input['gender'].lower() == 'male' else 0],
        'Temperature (C)': [temperature],
        'Humidity': [humidity],
        'Wind Speed (km/h)': [wind_speed],
    }

    symptom_vector = map_symptoms(user_input, symptom_columns)
    for col, val in zip(symptom_columns, symptom_vector):
        input_data[col] = [val]

    input_df = pd.DataFrame(input_data)
    
    # Creating lag features
    input_df['Temperature_lag1'] = input_df['Temperature (C)'].shift(1).fillna(input_df['Temperature (C)'].mean())
    input_df['Humidity_lag1'] = input_df['Humidity'].shift(1).fillna(input_df['Humidity'].mean())
    input_df['WindSpeed_lag1'] = input_df['Wind Speed (km/h)'].shift(1).fillna(input_df['Wind Speed (km/h)'].mean())

    model_features = model.feature_names_in_
    input_df = input_df[model_features]
    logger.info("Input data prepared for prediction.")

    # Prediction
    proba = model.predict_proba(input_df)[0]
    max_proba = max(proba)
    pred_index = proba.argmax()
    predicted_disease = label_encoder.inverse_transform([pred_index])[0]

    if show_top_n:
        top_indices = np.argsort(proba)[-3:][::-1]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_probs = [round(p * 100, 2) for p in proba[top_indices]]
        logger.info("Returning top 3 predictions with probabilities.")
        return "\n".join([f"{disease}: {p}%" for disease, p in zip(top_labels, top_probs)])

    # Confidence Handling 
    result = f"Predicted Disease: {predicted_disease} (Confidence: {max_proba:.2f})"

    if max_proba < 0.8:
        logger.warning(f"Prediction confidence ({max_proba:.2f}) is below threshold (0.8).")
        # Append the specific warning message
        result += "\ Confidence is low. It is highly recommended to consult a healthcare provider for confirmation."
    else:
        logger.info(f"Predicted disease: {predicted_disease} with confidence: {max_proba:.2f}")

    return result
