import os
import re
import json
import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers
from .weather_fetcher import fetch_and_log_weather_data  

logger = logging.getLogger("DLService")

class DLService:
    """
    Deep Learning (DL) service for disease prediction.
    Uses the same preprocessing, scaler, and model structure as the original DL module,
    while supporting real-time weather data retrieval via OpenWeatherMap API.
    """

    # Paths to pretrained model and artifacts
    MODEL_DIR = r"D:\brototype\week27\DL\DL_module\dl_codes\models"
    MODEL_PATH = os.path.join(MODEL_DIR, "dl_model.keras")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
    TEXT_CONFIG_PATH = os.path.join(MODEL_DIR, "text_vectorizer_config.json")
    TEXT_VOCAB_PATH = os.path.join(MODEL_DIR, "text_vectorizer_vocab.txt")

    def __init__(self, openweather_api_key: str):
        """Initialize DLService with pretrained model, scaler, and text vectorizer."""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"DL model not found at {self.MODEL_PATH}")

        # Load trained model
        tf.get_logger().setLevel(logging.ERROR)
        self.model = tf.keras.models.load_model(self.MODEL_PATH)
        tf.get_logger().setLevel(logging.INFO)
        logger.info(f"Model loaded successfully from: {self.MODEL_PATH}")

        # Load scaler
        self.scaler = joblib.load(self.SCALER_PATH)
        logger.info("Scaler loaded successfully.")

        # Load label encoder
        self.label_encoder = joblib.load(self.LABEL_ENCODER_PATH)
        logger.info("Label encoder loaded successfully.")

        # Load text vectorizer
        with open(self.TEXT_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.pop("vocabulary", None)
        self.text_vectorizer = layers.TextVectorization.from_config(config)
        with open(self.TEXT_VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.text_vectorizer.set_vocabulary(vocab)
        logger.info("Text vectorizer loaded successfully.")

        # Store API key for later use
        self.openweather_api_key = openweather_api_key

    # Helper Methods 

    def _normalize_text(self, text: str) -> str:
        """Clean and normalize user text before vectorization."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
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
            "knee ache": "knee_ache",
        }
        for phrase, token in replacements.items():
            text = text.replace(phrase, token)
        return text.strip()

    def _extract_city(self, text: str) -> str:
        """Extract city name from input text."""
        match = re.search(r"living in ([A-Za-z\s]+?)(?:\s+(?:and|with|but|while|suffering|having|experiencing|,|\.|$))", text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"

    def _extract_age(self, text: str) -> int:
        """Extract user age from text."""
        match = re.search(r"I am (\d+)\s+years? old", text, re.IGNORECASE)
        return int(match.group(1)) if match else 30

    # Prediction #

    async def predict(self, user_text: str) -> dict:
        """Predict disease using user symptoms + real-time weather features."""
        try:
            city = self._extract_city(user_text)
            age = self._extract_age(user_text)
            processed_text = self._normalize_text(user_text)

            # Fetch live weather data asynchronously
            weather_data = await fetch_and_log_weather_data(
                city, self.openweather_api_key, user_text, service_type="DL"
            )

            if "error" in weather_data:
                return {
                    "error": weather_data["error"],
                    "prediction": "N/A",
                    "status": "Failed to retrieve weather data."
                }

            # Build numeric feature array
            temp_c = weather_data.get("temp", 25.0)
            humidity = weather_data.get("humidity", 70.0) / 100  # normalize 0–1
            wind_speed = weather_data.get("wind_speed", 10.0)
            symptom_count = len(processed_text.split())

            numeric_features = np.array(
                [[age, temp_c, humidity, wind_speed, symptom_count]],
                dtype=np.float32
            )

            # Scale numeric features
            X_numeric_scaled = self.scaler.transform(numeric_features)

            # Vectorize text
            X_text_seq = self.text_vectorizer(np.array([processed_text])).numpy()

            # Predict (use named input dict to match training)
            predictions = self.model.predict(
                {"numeric_input": X_numeric_scaled, "text_input": X_text_seq},
                verbose=0
            )

            predicted_idx = np.argmax(predictions, axis=1)[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence = float(np.max(predictions))

            # Top-5 probabilities
            class_probs = predictions[0]
            top5 = sorted(
                zip(self.label_encoder.classes_, class_probs),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "top_5": [{label: float(prob)} for label, prob in top5],
                "raw_weather_data": {k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))},
                "status": "Success",
                "note": user_text,
                "city": city,
            }

        except Exception as e:
            logger.error(f"DL prediction failed: {e}", exc_info=True)
            return {"error": str(e), "status": "Error", "prediction": "N/A"}



# TEST SECTION 
import os
import asyncio
import json

# if __name__ == "__main__":
#     async def test_dl_service():
#         """Run a simple test prediction for verification."""
#         # Read from environment variable (safe)
#         openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")

#         if not openweather_api_key:
#             print("Missing environment variable: OPENWEATHER_API_KEY")
#             print("Please set it before running the script.")
#             return

#         try:
#             dl_service = DLService(openweather_api_key=openweather_api_key)

#             user_input = "I am 35 years old living in Calicut. I am feeling severe headache, chest pain and dizziness."
#             result = await dl_service.predict(user_input)

#             print("\n=== DLService Test Result ===")
#             print(json.dumps(result, indent=4))

#         except Exception as e:
#             print(f"Error during test: {e}")

#     asyncio.run(test_dl_service())
