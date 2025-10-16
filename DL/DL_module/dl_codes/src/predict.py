import os
import logging
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from preprocess import preprocess_user_input

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = r"D:\brototype\week27\DL\DL_module\dl_codes\models"
MODEL_PATH = os.path.join(MODEL_DIR, "dl_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
TEXT_VOCAB_PATH = os.path.join(MODEL_DIR, "text_vectorizer_vocab.txt")
TEXT_CONFIG_PATH = os.path.join(MODEL_DIR, "text_vectorizer_config.json")


def load_preprocessors():
    """
    Load the scaler, text vectorizer, and label encoder from the local model directory.

    Returns:
        tuple: (scaler, text_vectorizer, label_encoder)
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded successfully.")

        with open(TEXT_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        text_vectorizer = TextVectorization.from_config(config)

        with open(TEXT_VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        text_vectorizer.set_vocabulary(vocab)
        logger.info("TextVectorization layer loaded successfully.")

        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("LabelEncoder loaded successfully.")

        return scaler, text_vectorizer, label_encoder

    except Exception as e:
        logger.error(f"Error loading preprocessors: {e}")
        return None, None, None


def predict_disease(user_full_symptoms: str):
    """
    Predict the most likely disease based on user symptoms.

    Args:
        user_full_symptoms (str): A comma-separated list of user symptoms.

    Returns:
        tuple: (predicted_disease, confidence_scores)
               predicted_disease (str) – the top predicted class
               confidence_scores (dict) – probability for each class
    """
    scaler, text_vectorizer, label_encoder = load_preprocessors()
    if not all([scaler, text_vectorizer, label_encoder]):
        return "Prediction failed: Could not load preprocessors.", None

    try:
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded from: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return "Prediction failed: Could not load the model.", None

    try:
        normalized_symptoms = " ".join([sym.strip().lower() for sym in user_full_symptoms.split(",")])
        synonym_map = {
            "head pain": "headache",
            "nausea feeling": "nausea",
            "dizzy": "dizziness",
            "light headed": "lightheadedness",
            "joint ache": "joint_pain"
        }
        for key, value in synonym_map.items():
            normalized_symptoms = normalized_symptoms.replace(key, value)

        X_numeric_scaled, X_text_sequenced, _ = preprocess_user_input(
            normalized_symptoms, scaler, text_vectorizer
        )

    except Exception as e:
        logger.error(f"Failed to preprocess input: {e}")
        return "Prediction failed: Error during input processing.", None

    try:
        predictions = model.predict({
            'numeric_input': X_numeric_scaled,
            'text_input': X_text_sequenced
        })

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]

        class_probabilities = predictions[0]
        confidence_scores = {
            label: float(prob) for label, prob in zip(label_encoder.classes_, class_probabilities)
        }

        return predicted_disease, confidence_scores

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Prediction failed during inference.", None


if __name__ == "__main__":
    """
    Run a simple test prediction with sample input.
    """
    user_input = "headache, nausea, fever"
    prediction, scores = predict_disease(user_input)

    print("\nUser Input:", user_input)
    print("Predicted Disease:", prediction)

    if scores:
        print("\nTop 5 Confidence Scores:")
        for label, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{label}: {prob:.4f}")
