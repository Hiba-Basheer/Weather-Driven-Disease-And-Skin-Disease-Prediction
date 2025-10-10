"""
DL module for predicting diseases from unstructured text input.
"""

import os
import logging
import tensorflow as tf
from src.preprocess import preprocess_dl_text
from src.constants import LABEL_MAP  # Shared label mapping for DL and ML

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to trained DL model
MODEL_PATH = r"D:\brototype\week27\RAG_chatbot\models\dl_model.keras"

# Load DL model
try:
    dl_model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("DL model loaded successfully.")
except Exception as e:
    dl_model = None
    logging.error(f"Failed to load DL model: {e}")

def predict_from_text(text: str) -> str:
    """
    Predict disease from unstructured symptom description using the trained DL model.

    Parameters:
        text (str): Free-form symptom description provided by the user.

    Returns:
        str: Predicted disease label or an error message if prediction fails.
    """
    if dl_model is None:
        logging.warning("DL model is not available.")
        return "Model not available."

    try:
        # Preprocess the input text using the pipeline
        processed = preprocess_dl_text(text)

        # Run prediction
        prediction = dl_model.predict(processed)

        # Get predicted label from index
        predicted_index = prediction.argmax(axis=1)[0]
        predicted_label = LABEL_MAP.get(predicted_index, "Unknown condition")

        logging.info(f"DL prediction: {predicted_label}")
        return predicted_label
    except Exception as e:
        logging.error(f"Error during DL prediction: {e}")
        return "Prediction failed due to preprocessing error."