import os
import joblib
import logging
import pandas as pd
import numpy as np
from .weather_fetcher import fetch_and_log_weather_data

logger = logging.getLogger("MLService")

class MLService:
    """
    Service class for ML predictions using trained model + live weather.
    """

    FEATURE_NAMES_FILE = "ml_expected_columns.pkl"
    LABEL_ENCODER_FILE = "label_encoder.pkl"
    MODEL_FILE = "trained_model.pkl"

    def __init__(self, model_dir: str = None, api_key: str = None):
        """
        Initializes MLService.

        model_dir: path to folder containing trained_model.pkl, label_encoder.pkl, ml_expected_columns.pkl
        api_key: OpenWeatherMap API key
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "ml")
        self.model_dir = model_dir
        self.api_key = api_key

        # Load model
        model_path = os.path.join(model_dir, self.MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)

        # Load label encoder
        label_path = os.path.join(model_dir, self.LABEL_ENCODER_FILE)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label encoder not found at {label_path}")
        self.label_encoder = joblib.load(label_path)

        # Load expected features
        features_path = os.path.join(model_dir, self.FEATURE_NAMES_FILE)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Expected features file not found at {features_path}")
        self.expected_features = joblib.load(features_path)

        logger.info(f"Model, encoder, and expected features loaded successfully from {model_dir}")

    async def predict(self, user_input: dict) -> dict:
        """
        Predict disease using structured input + live weather.

        user_input should contain: {"age": int, "gender": str, "city": str, "symptoms": str}
        """
        try:
            city = user_input.get("city")
            if self.api_key is None:
                raise ValueError("OpenWeatherMap API key not set for weather fetching.")

            # Fetch live weather
            weather_data = await fetch_and_log_weather_data(city, self.api_key, user_input.get("symptoms"), service_type="ML")
            if 'error' in weather_data:
                return {
                    "error": weather_data['error'],
                    "prediction": "N/A",
                    "status": "Failed to retrieve weather data."
                }

            # Prepare feature dictionary
            features_dict = {col: 0 for col in self.expected_features}

            # Fill numeric/weather features
            features_dict["Age"] = user_input.get("age", 0)
            features_dict["Temperature (C)"] = weather_data.get("temp", 30)
            features_dict["Humidity"] = weather_data.get("humidity", 70)
            features_dict["Wind Speed (km/h)"] = weather_data.get("wind_speed", 5)

            # Map user symptoms to feature columns
            user_symptoms = [s.strip().lower() for s in user_input.get("symptoms", "").split(",")]
            for symptom in user_symptoms:
                if symptom in features_dict:
                    features_dict[symptom] = 1

            # Gender as numeric/categorical if exists in expected_features
            gender = user_input.get("gender", "").lower()
            if "gender" in self.expected_features:
                features_dict["gender"] = 1 if gender == "male" else 0

            # Create DataFrame
            df_final = pd.DataFrame([features_dict])

            # Predict
            X_input = df_final.values
            pred_proba = self.model.predict_proba(X_input)[0]
            pred_class_index = np.argmax(pred_proba)
            predicted_label = self.label_encoder.inverse_transform([pred_class_index])[0]

            return {
                "prediction": predicted_label,
                "confidence": float(np.max(pred_proba)),
                "raw_weather_data": {k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))},
                "status": "Success",
                "user_info": user_input
            }

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {
                "error": str(e),
                "prediction": "N/A",
                "status": "Error"
            }
