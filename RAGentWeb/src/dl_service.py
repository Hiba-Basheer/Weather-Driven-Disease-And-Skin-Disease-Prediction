import os
import re
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .weather_fetcher import fetch_and_log_weather_data

logger = logging.getLogger("DLService")


class DLService:
    """
    Deep Learning (DL) service for disease prediction.
    Combines user free-text input with age and weather features.
    """

    # Local artifact paths
    BASE_DIR = r"D:\brototype\week27\RAGentWeb\models\dl"
    MODEL_PATH = os.path.join(BASE_DIR, "dl_model.keras")
    VECTORIZER_CONFIG_PATH = os.path.join(BASE_DIR, "text_vectorizer_config.json")
    VECTORIZER_VOCAB_PATH = os.path.join(BASE_DIR, "text_vectorizer_vocab.txt")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

    DISEASE_LABELS = [
        "Heart Attack",
        "Influenza",
        "Dengue",
        "Sinusitis",
        "Eczema",
        "Common Cold",
        "Heat Stroke",
        "Migraine",
        "Malaria"
    ]

    def __init__(self, api_key: str):
        """
        Initialize the DLService with pretrained model and text vectorizer.

        Args:
            api_key (str): OpenWeather API key.
        """
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"DL model not found at: {self.MODEL_PATH}")

        # Suppress TF logs during model loading
        tf.get_logger().setLevel(logging.ERROR)
        self.model = tf.keras.models.load_model(self.MODEL_PATH)
        tf.get_logger().setLevel(logging.INFO)
        logger.info(f"Model loaded successfully from: {self.MODEL_PATH}")

        # Load text vectorizer
        self.text_vectorizer = self._load_text_vectorizer_manual(
            self.VECTORIZER_CONFIG_PATH,
            self.VECTORIZER_VOCAB_PATH
        )

        if self.text_vectorizer:
            self.MAX_LEN = self.text_vectorizer.get_config().get("output_sequence_length", 50)
            logger.info(f"TextVectorization layer initialized with MAX_LEN={self.MAX_LEN}.")
        else:
            self.MAX_LEN = 50
            logger.warning("Text vectorizer failed to initialize. DL prediction may be unreliable.")

        self.api_key = api_key

    # Internal Utilities
    def _load_text_vectorizer_manual(self, config_path: str, vocab_path: str):
        """
        Load TextVectorization layer from config and vocabulary files.

        Args:
            config_path (str): Path to the JSON configuration file.
            vocab_path (str): Path to the vocabulary text file.

        Returns:
            TextVectorization: Reconstructed text vectorizer layer.
        """
        if not os.path.exists(config_path) or not os.path.exists(vocab_path):
            logger.error(f"Vectorizer config/vocab missing at: {config_path} or {vocab_path}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config.pop("vocabulary", None)

            vectorizer = layers.TextVectorization.from_config(config)
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
            vectorizer.set_vocabulary(vocab)

            return vectorizer
        except Exception as e:
            logger.error(f"Failed to load TextVectorization layer: {e}")
            return None

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess input text to align with training data.

        Args:
            text (str): Raw user input text.

        Returns:
            str: Cleaned and normalized text.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)

        replacements = {
            "joint pain": "joint_pain",
            "abdominal pain": "abdominal_pain",
            "high fever": "high_fever",
            "runny nose": "runny_nose",
            "pain behind the eyes": "pain_behind_the_eyes",
            "chest pain": "chest_pain",
            "shortness of breath": "shortness_of_breath",
            "skin irritation": "skin_irritation",
            "throbbing headache": "throbbing_headache",
            "back pain": "back_pain",
            "knee ache": "knee_ache"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        return text

    # Data Extraction
    def extract_city_from_text(self, text: str) -> str:
        """Extract city name from input text."""
        match = re.search(r"living in ([A-Za-z\s]+)", text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"

    def extract_age_from_text(self, text: str) -> int:
        """Extract age value from input text."""
        match = re.search(r"I am (\d+)\s+years? old", text, re.IGNORECASE)
        return int(match.group(1)) if match else 30

    # Main Prediction Method
    async def predict(self, user_text: str) -> dict:
        """
        Predict disease from user symptoms and weather data.

        Args:
            user_text (str): User-provided input describing symptoms and location.

        Returns:
            dict: Prediction result with confidence and metadata.
        """
        city = self.extract_city_from_text(user_text)
        age = self.extract_age_from_text(user_text)
        processed_text = self._preprocess_text(user_text)

        # Fetch live weather data
        weather_data = await fetch_and_log_weather_data(city, self.api_key, user_text, service_type="DL")
        if "error" in weather_data:
            return {
                "error": weather_data["error"],
                "prediction": "N/A",
                "status": "Failed to retrieve weather data."
            }

        if not self.text_vectorizer:
            return {
                "error": "Text vectorizer not initialized. DL prediction disabled.",
                "prediction": "N/A",
                "status": "Error"
            }

        try:
            # Vectorize text
            padded_text_input = self.text_vectorizer([processed_text]).numpy()

            # Prepare structured features
            structured_features = np.array([
                age,
                weather_data.get("temp", 0),
                weather_data.get("humidity", 0),
                weather_data.get("wind_speed", 0)
            ], dtype=np.float32).reshape(1, -1)

            # Ensure structured_features has 5 columns
            if structured_features.shape[1] < 5:
                missing = 5 - structured_features.shape[1]
                structured_features = np.pad(structured_features, ((0, 0), (0, missing)), mode='constant')
            # Model prediction
            prediction_proba = self.model.predict([structured_features, padded_text_input])[0]
            predicted_index = np.argmax(prediction_proba)
            prediction_label = (
                self.DISEASE_LABELS[predicted_index]
                if 0 <= predicted_index < len(self.DISEASE_LABELS)
                else f"Unknown Class {predicted_index}"
            )

            return {
                "prediction": prediction_label,
                "confidence": float(np.max(prediction_proba)),
                "raw_weather_data": {
                    k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))
                },
                "status": "Success",
                "note": user_text,
                "city": city
            }

        except Exception as e:
            logger.error(f"DL prediction error for '{user_text}': {e}", exc_info=True)
            return {"error": str(e), "prediction": "N/A", "status": "Error"}
