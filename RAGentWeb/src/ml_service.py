"""
ml_service.py
Machine Learning–based disease prediction using structured input + live weather.

This module defines the MLService class, which loads a trained scikit-learn model
to predict diseases from structured user inputs (age, gender, city, symptoms),
enriched with live weather data fetched from OpenWeatherMap.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from .weather_fetcher import fetch_and_log_weather_data

logger = logging.getLogger("MLService")


class MLService:
    """
    Machine Learning Disease Prediction Service.

    Loads a trained scikit-learn model, label encoder, and expected feature list.
    Accepts structured input (age, gender, city, symptoms), fetches live weather
    for the specified city, constructs a feature vector, and predicts the most
    probable disease with confidence.
    """

    FEATURE_NAMES_FILE = "ml_expected_columns.pkl"
    LABEL_ENCODER_FILE = "label_encoder.pkl"
    MODEL_FILE = "trained_model.pkl"

    def __init__(self, model_dir: str = None, api_key: str = None):
        """
        Initialize the MLService instance.

        Args:
            model_dir (str, optional): Directory containing model and related files.
                Defaults to ../models/ml relative to the current script.
            api_key (str, optional): OpenWeatherMap API key for weather fetching.

        Raises:
            FileNotFoundError: If the model, label encoder, or expected features file is missing.
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

        # Load expected feature names
        features_path = os.path.join(model_dir, self.FEATURE_NAMES_FILE)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Expected features file not found at {features_path}")
        self.expected_features = joblib.load(features_path)

        logger.info(f"Model, encoder, and expected features loaded successfully from {model_dir}")

    async def predict(self, user_input: dict) -> dict:
        """
        Predict a disease using structured input and live weather data.

        Args:
            user_input (dict): Dictionary containing structured fields:
                {
                    "age": int,
                    "gender": str,
                    "city": str,
                    "symptoms": str  # comma-separated symptom list
                }

        Returns:
            dict: Prediction results containing:
                - prediction (str): Predicted disease name.
                - confidence (float): Model confidence score.
                - raw_weather_data (dict): Live weather info used in prediction.
                - user_info (dict): Echo of the provided user input.
                - status (str): "Success" or "Error".
                - error (optional, str): Error message if any failure occurred.
        """
        try:
            city = user_input.get("city")
            if self.api_key is None:
                raise ValueError("OpenWeatherMap API key not set for weather fetching.")

            # Fetch live weather
            weather_data = await fetch_and_log_weather_data(
                city,
                self.api_key,
                user_input.get("symptoms"),
                service_type="ML"
            )

            # Weather fetch failed
            if 'error' in weather_data:
                return {
                    "error": weather_data['error'],
                    "prediction": "N/A",
                    "status": "Failed to retrieve weather data."
                }

            # Prepare feature dictionary initialized to zero
            features_dict = {col: 0 for col in self.expected_features}

            # Fill numeric/weather features
            features_dict["Age"] = user_input.get("age", 0)
            features_dict["Temperature (C)"] = weather_data.get("temp", 30)
            features_dict["Humidity"] = weather_data.get("humidity", 70)
            features_dict["Wind Speed (km/h)"] = weather_data.get("wind_speed", 5)

            # Map symptoms to feature columns
            user_symptoms = [s.strip().lower() for s in user_input.get("symptoms", "").split(",")]
            for symptom in user_symptoms:
                if symptom in features_dict:
                    features_dict[symptom] = 1

            # Encode gender if part of expected features
            gender = user_input.get("gender", "").lower()
            if "gender" in self.expected_features:
                features_dict["gender"] = 1 if gender == "male" else 0

            # Create DataFrame for model input
            df_final = pd.DataFrame([features_dict])

            # Predict probabilities and most likely class
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


# EVALUATION BLOCK
if __name__ == "__main__":
    """
    Evaluation script for MLService.

    This block performs validation of the machine learning prediction service using
    predefined test cases. It measures accuracy against known expected outcomes.

    Each test case provides:
      - age
      - gender
      - city
      - symptoms
      - expected disease label

    Results:
      - Displays predicted vs. expected disease
      - Reports total accuracy across test set
    """
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHERMAP_API_KEY not found")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = MLService(api_key=api_key)

    test_cases = [
        ({"age": 35, "gender": "male", "city": "Calicut", "symptoms": "high_fever, joint_pain, pain_behind_the_eyes"}, "Dengue"),        
        ({"age": 25, "gender": "female", "city": "Kochi", "symptoms": "high_fever, chills, headache, shivering"}, "Malaria"),
        ({"age": 28, "gender": "female", "city": "Bangalore", "symptoms": "cough, runny_nose, headache, fatigue"}, "Common Cold"),
        ({"age": 45, "gender": "male", "city": "Chennai", "symptoms": "chest_pain, dizziness, vomiting, nausea"}, "Heart Attack"),        
        ({"age": 30, "gender": "female", "city": "Delhi", "symptoms": "headache, nausea, dizziness"}, "Migraine"),
    ]

    correct = 0
    total = len(test_cases)

    for i, (user_input, expected) in enumerate(test_cases, 1):
        result = asyncio.run(service.predict(user_input))
        pred = result.get("prediction", "N/A")
        conf = result.get("confidence", 0.0)

        logger.info("Test %d: %s", i, user_input)
        logger.info("Prediction: %s (%.2f) | Expected: %s", pred, conf, expected)
        if pred == expected:
            correct += 1
            logger.info("Status: PASSED")
        else:
            logger.info("Status: FAILED")

    accuracy = correct / total * 100
    logger.info("Accuracy: %d/%d (%.1f%%)", correct, total, accuracy)
