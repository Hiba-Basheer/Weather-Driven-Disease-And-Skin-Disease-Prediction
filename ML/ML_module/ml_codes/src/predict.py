import pandas as pd
import numpy as np
import logging
from utils.preprocess import preprocess_data
from utils.weather_api import get_weather_data
from symptom_mapper import map_symptoms
from load_model import load_trained_model

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_symptoms(user_symptoms, valid_columns):
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


def predict_disease(user_input, model_path, le_path, symptom_columns, show_top_n=False):
    """
    Predict disease based on user symptoms and weather data, with confidence handling.

    Args:
        user_input (dict): Dictionary with keys - age, gender, symptoms, and optionally city or raw weather values.
        model_path (str): Path to the trained model file.
        le_path (str): Path to the label encoder file.
        symptom_columns (list): List of known symptom feature columns.
        show_top_n (bool): If True, return top 3 most probable predictions.

    Returns:
        str: Prediction result.
    """
    logger.info("Starting disease prediction...")
    
    user_input['symptoms'] = normalize_symptoms(user_input['symptoms'], symptom_columns)
    model, label_encoder = load_trained_model(model_path, le_path)
    logger.info("Model and label encoder loaded successfully.")

    if 'city' in user_input:
        weather = get_weather_data(user_input['city'])
        logger.info(f"Fetched weather data for city: {user_input['city']}")
        temperature = weather['temperature']
        humidity = weather['humidity']
        wind_speed = weather['wind_speed']
    else:
        temperature = user_input.get('temperature')
        humidity = user_input.get('humidity')
        wind_speed = user_input.get('wind_speed')
        logger.info("Using raw weather data from user input.")

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
    input_df['Temperature_lag1'] = input_df['Temperature (C)'].shift(1).fillna(input_df['Temperature (C)'].mean())
    input_df['Humidity_lag1'] = input_df['Humidity'].shift(1).fillna(input_df['Humidity'].mean())
    input_df['WindSpeed_lag1'] = input_df['Wind Speed (km/h)'].shift(1).fillna(input_df['Wind Speed (km/h)'].mean())

    model_features = model.feature_names_in_
    input_df = input_df[model_features]
    logger.info("Input data prepared for prediction.")

    proba = model.predict_proba(input_df)[0]
    max_proba = max(proba)
    pred_index = proba.argmax()

    if show_top_n:
        top_indices = np.argsort(proba)[-3:][::-1]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_probs = [round(p * 100, 2) for p in proba[top_indices]]
        logger.info("Returning top 3 predictions with probabilities.")
        return "\n".join([f"{disease}: {p}%" for disease, p in zip(top_labels, top_probs)])

    if max_proba < 0.8:
        logger.warning("Prediction confidence is below threshold.")
        return "Prediction confidence is low. Please consult a healthcare provider."

    predicted_disease = label_encoder.inverse_transform([pred_index])[0]
    logger.info(f"Predicted disease: {predicted_disease} with confidence: {max_proba:.2f}")
    return predicted_disease
