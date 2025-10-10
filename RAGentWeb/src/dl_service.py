import os
import re
import json
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras import layers
from .weather_fetcher import fetch_and_log_weather_data

logger = logging.getLogger("DLService")

class DLService:
    """
    Service class for Deep Learning (DL) prediction.
    Accepts a single free-text input from the user, extracts age, city, 
    fetches weather data, and predicts disease risk.
    """
    
    VECTORIZER_CONFIG_FILE = "text_vectorizer_config.json" 
    VECTORIZER_VOCAB_FILE = "text_vectorizer_vocab.txt" 

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
    
    def __init__(self, model_path: str, api_key: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DL model not found at: {model_path}")

        tf.get_logger().setLevel(logging.ERROR)
        self.model = tf.keras.models.load_model(model_path)
        tf.get_logger().setLevel(logging.INFO)

        model_dir = os.path.dirname(model_path)
        
        # 1. Load Text Vectorizer
        self.text_vectorizer = self._load_text_vectorizer_manual(
            os.path.join(model_dir, self.VECTORIZER_CONFIG_FILE),
            os.path.join(model_dir, self.VECTORIZER_VOCAB_FILE)
        )
        
        if self.text_vectorizer:
            self.MAX_LEN = self.text_vectorizer.get_config().get('output_sequence_length', 50)
            logger.info(f"TextVectorization layer successfully initialized with MAX_LEN={self.MAX_LEN}.")
        else:
            self.MAX_LEN = 50 
            logger.warning("Text vectorizer initialization failed. DL prediction will be disabled/unreliable.")
            
        self.api_key = api_key
        self.feature_names = ['age', 'temp', 'humidity', 'wind_speed'] 
        logger.info(f"DL Keras Model loaded from {model_path}.")

    def _load_text_vectorizer_manual(self, config_path: str, vocab_path: str):
        """Loads TextVectorization layer manually from its config and vocabulary file."""
        if not os.path.exists(config_path) or not os.path.exists(vocab_path):
            logger.error(f"Vectorizer config/vocab files missing. Config: {os.path.exists(config_path)}, Vocab: {os.path.exists(vocab_path)}")
            return None
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                layer_config = json.load(f)
            
            if 'vocabulary' in layer_config:
                 layer_config.pop('vocabulary') 

            vectorizer = layers.TextVectorization.from_config(layer_config)

            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocabulary = [line.strip() for line in f if line.strip()]

            vectorizer.set_vocabulary(vocabulary)
            return vectorizer
        
        except Exception as e:
            logger.error(f"Failed to load TextVectorization layer manually: {e}")
            return None
    
    def extract_city_from_text(self, text: str) -> str:
        """Extracts the city name from the user's text input."""
        match = re.search(r'living in ([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            logger.info(f"City extracted from text: {city}")
            return city
        logger.warning("City not found in text input. Defaulting to 'Unknown'.")
        return "Unknown"
    
    def extract_age_from_text(self, text: str) -> int:
        """Extracts age from text, defaulting to 30 if not found."""
        match = re.search(r'I am (\d+)\s+years? old', text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            return age
        logger.warning("Age not found in text input. Defaulting to 30.")
        return 30


    async def predict(self, user_text: str) -> dict:
        city = self.extract_city_from_text(user_text)
        age = self.extract_age_from_text(user_text)

        weather_data = await fetch_and_log_weather_data(city, self.api_key, user_text, service_type='DL')

        if 'error' in weather_data:
            return {"error": weather_data['error'], "prediction": "N/A", "status": "Failed to retrieve weather data."}

        try:
            if not self.text_vectorizer:
                return {"error": "Text vectorizer not initialized. DL prediction is disabled.", "prediction": "N/A", "status": "Error"}

            # PREPARE INPUTS
            
            # Prepare Text Input
            padded_text_input = self.text_vectorizer([user_text]).numpy() 

            # Prepare Structured (Numerical) Input
            numerical_features = [
                age,
                weather_data.get('temp', 0),
                weather_data.get('humidity', 0),
                weather_data.get('wind_speed', 0)
            ]
            structured_data_input = np.array(numerical_features, dtype=np.float32).reshape(1, -1) 
            
            # PERFORM PREDICTION
            
            prediction_proba = self.model.predict([structured_data_input, padded_text_input])[0]
            predicted_index = np.argmax(prediction_proba)
            
            # Get the label name using the correct index lookup
            if 0 <= predicted_index < len(self.DISEASE_LABELS):
                prediction_label = self.DISEASE_LABELS[predicted_index]
            else:
                prediction_label = f"Unknown Class {predicted_index}"

            # Format Results
            return {
                "prediction": str(prediction_label),
                "confidence": float(np.max(prediction_proba)),
                "raw_weather_data": {k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))},
                "status": "Success",
                "note": user_text,
                "city": city
            }

        except Exception as e:
            logger.error(f"DL prediction error for text '{user_text}': {e}", exc_info=True)
            return {"error": f"Internal prediction processing error: {e}", "prediction": "N/A", "status": "Error"}
