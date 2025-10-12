"""
ML module for predicting weather-driven diseases from structured input.
Accepts character-based features and returns a disease prediction.
"""

import joblib
import os
import logging
from src.constants import LABEL_MAP  

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to trained ML model
MODEL_PATH = r"D:\brototype\week27\RAG_chatbot\models\trained_model.pkl"

# Load ML model
try:
    ml_model = joblib.load(MODEL_PATH)
    logging.info("ML model loaded successfully.")
except Exception as e:
    ml_model = None
    logging.error(f"Failed to load ML model: {e}")

def predict_from_structured(features: list) -> str:
    """
    Predict disease from structured character-based input using the trained ML model.

    Parameters:
        features (list): List of categorical or textual features,
                         ordered exactly as expected by the model.

    Returns:
        str: Predicted disease name, or error message if prediction fails.
    """
    if ml_model is None:
        logging.warning("ML model is not available.")
        return "Model not available."

    try:
        prediction = ml_model.predict([features])[0]

        # If model returns index, map it to label
        if isinstance(prediction, int):
            predicted_label = LABEL_MAP.get(prediction, "Unknown condition")
        else:
            predicted_label = str(prediction)

        logging.info(f"ML prediction: {predicted_label}")
        return predicted_label
    except Exception as e:
        logging.error(f"Error during ML prediction: {e}")
        return "Prediction failed due to input error."