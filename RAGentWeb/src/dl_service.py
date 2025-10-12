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
        
        # Load Text Vectorizer
        self.text_vectorizer = self._load_text_vectorizer_manual(
            os.path.join(model_dir, self.VECTORIZER_CONFIG_FILE),
            os.path.join(model_dir, self.VECTORIZER_VOCAB_FILE)
        )
        
        if self.text_vectorizer:
            self.MAX_LEN = self.text_vectorizer.get_config().get('output_sequence_length', 50)
            logger.info(f"TextVectorization layer initialized with MAX_LEN={self.MAX_LEN}.")
        else:
            self.MAX_LEN = 50 
            logger.warning("Text vectorizer failed to initialize. DL prediction may be unreliable.")
            
        self.api_key = api_key

    def _load_text_vectorizer_manual(self, config_path: str, vocab_path: str):
        """Load TextVectorization layer from config + vocabulary files."""
        if not os.path.exists(config_path) or not os.path.exists(vocab_path):
            logger.error(f"Vectorizer config/vocab files missing. Config: {os.path.exists(config_path)}, Vocab: {os.path.exists(vocab_path)}")
            return None
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                layer_config = json.load(f)
            layer_config.pop('vocabulary', None)
            vectorizer = layers.TextVectorization.from_config(layer_config)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f if line.strip()]
            vectorizer.set_vocabulary(vocab)
            return vectorizer
        except Exception as e:
            logger.error(f"Failed to load TextVectorization layer: {e}")
            return None

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text to match training:
        - Lowercase
        - Remove punctuation
        - Concatenate multi-word symptoms to match vocab
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  
        replacements = {
        "nausea": "nausea",
        "joint pain": "joint_pain",
        "abdominal pain": "abdominal_pain",
        "high fever": "high_fever",
        "chills": "chills",
        "fatigue": "fatigue",
        "runny nose": "runny_nose",
        "pain behind the eyes": "pain_behind_the_eyes",
        "dizziness": "dizziness",
        "headache": "headache",
        "chest pain": "chest_pain",
        "vomiting": "vomiting",
        "cough": "cough",
        "shivering": "shivering",
        "asthma history": "asthma_history",
        "high cholesterol": "high_cholesterol",
        "diabetes": "diabetes",
        "obesity": "obesity",
        "hiv/aids": "hiv_aids",
        "nasal polyps": "nasal_polyps",
        "asthma": "asthma",
        "high blood pressure": "high_blood_pressure",
        "severe headache": "severe_headache",
        "weakness": "weakness",
        "trouble seeing": "trouble_seeing",
        "fever": "fever",
        "body aches": "body_aches",
        "sore throat": "sore_throat",
        "sneezing": "sneezing",
        "diarrhea": "diarrhea",
        "rapid breathing": "rapid_breathing",
        "rapid heart rate": "rapid_heart_rate",
        "pain behind eyes": "pain_behind_eyes",
        "swollen glands": "swollen_glands",
        "rashes": "rashes",
        "sinus headache": "sinus_headache",
        "facial pain": "facial_pain",
        "shortness of breath": "shortness_of_breath",
        "reduced smell and taste": "reduced_smell_and_taste",
        "skin irritation": "skin_irritation",
        "itchiness": "itchiness",
        "throbbing headache": "throbbing_headache",
        "confusion": "confusion",
        "back pain": "back_pain",
        "knee ache": "knee_ache",
    }


        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def extract_city_from_text(self, text: str) -> str:
        """Extract city from text."""
        match = re.search(r'living in ([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            return city
        return "Unknown"
    
    def extract_age_from_text(self, text: str) -> int:
        """Extract age from text."""
        match = re.search(r'I am (\d+)\s+years? old', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 30  # default

    async def predict(self, user_text: str) -> dict:
        city = self.extract_city_from_text(user_text)
        age = self.extract_age_from_text(user_text)
        processed_text = self._preprocess_text(user_text)

        weather_data = await fetch_and_log_weather_data(city, self.api_key, user_text, service_type='DL')
        if 'error' in weather_data:
            return {"error": weather_data['error'], "prediction": "N/A", "status": "Failed to retrieve weather data."}

        if not self.text_vectorizer:
            return {"error": "Text vectorizer not initialized. DL prediction disabled.", "prediction": "N/A", "status": "Error"}

        try:
            # Vectorize text
            padded_text_input = self.text_vectorizer([processed_text]).numpy()
            
            # Prepare structured features
            structured_features = np.array([
                age,
                weather_data.get('temp', 0),
                weather_data.get('humidity', 0),
                weather_data.get('wind_speed', 0)
            ], dtype=np.float32).reshape(1, -1)

            # Predict
            prediction_proba = self.model.predict([structured_features, padded_text_input])[0]
            predicted_index = np.argmax(prediction_proba)
            prediction_label = self.DISEASE_LABELS[predicted_index] if 0 <= predicted_index < len(self.DISEASE_LABELS) else f"Unknown Class {predicted_index}"

            return {
                "prediction": prediction_label,
                "confidence": float(np.max(prediction_proba)),
                "raw_weather_data": {k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))},
                "status": "Success",
                "note": user_text,
                "city": city
            }
        except Exception as e:
            logger.error(f"DL prediction error for '{user_text}': {e}", exc_info=True)
            return {"error": str(e), "prediction": "N/A", "status": "Error"}
