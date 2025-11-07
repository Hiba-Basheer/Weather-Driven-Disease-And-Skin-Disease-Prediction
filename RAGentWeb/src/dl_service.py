"""
dl_service.py
Deep Learning-based disease prediction using free-text user input and live weather data.

It supports TextVectorization loading, input validation,
and robust parsing for symptom text.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict

import joblib
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras import layers

# Configure TensorFlow logging
tf.get_logger().setLevel("ERROR")

# Detect environment
IS_DOCKER = (
    os.path.exists("/.dockerenv") or os.getenv("DOCKER", "false").lower() == "true"
)
MODEL_ROOT = (
    "/app/models/dl"
    if IS_DOCKER
    else os.path.join(os.path.dirname(__file__), "..", "models", "dl")
)

logger = logging.getLogger("DLService")

# MODEL CONSTANTS
SYMPTOM_COLUMNS = [
    "chills",
    "loss_of_balance",
    "reduced_smell_and_taste",
    "nasal_polyps",
    "pain_behind_eyes",
    "pain_radiating_to_left_arm",
    "dry_skin",
    "joint_pain",
    "body_aches",
    "upper_back_pain",
    "joint_stiffness",
    "knee_ache",
    "slurred_speech",
    "back_pain",
    "arm_pain",
    "rapid_heart_rate",
    "rapid_breathing",
    "throbbing_headache",
    "diarrhea",
    "high_cholesterol",
    "vomiting",
    "sinus_headache",
    "sweating",
    "lightheadedness",
    "shortness_of_breath",
    "sore_throat",
    "trouble_seeing",
    "sensitivity_to_light",
    "trouble_speaking",
    "weakness_in_arms_or_legs",
    "swollen_glands",
    "fever",
    "anxiety",
    "jaw_pain",
    "high_fever",
    "weakness",
    "nausea",
    "dizziness",
    "abdominal_pain",
    "runny_nose",
    "sneezing",
    "sensitivity_to_sound",
    "sudden_numbness_on_one_side",
    "facial_pain",
    "bleeding_gums",
    "fatigue",
    "loss_of_appetite",
    "skin_irritation",
    "headache",
    "severe_headache",
    "confusion",
    "chest_pain",
    "diabetes",
    "cough",
    "hiv_aids",
    "asthma_history",
    "itchiness",
    "asthma",
    "jaw_discomfort",
    "high_blood_pressure",
    "blurred_vision",
    "rashes",
    "obesity",
    "shivering",
    "pain_behind_the_eyes",
]

TOTAL_BINARY_FEATURES = len(SYMPTOM_COLUMNS) + 1  # 65 + 1 (gender)
EXPECTED_NUMERIC_SHAPE = (None, 5 + TOTAL_BINARY_FEATURES)  # (None, 71)
EXPECTED_TEXT_SHAPE = (None, 100)


class DLService:
    """Deep Learning service for disease prediction."""

    MODEL_DIR = MODEL_ROOT
    MODEL_PATH = os.path.join(MODEL_DIR, "dl_model.keras")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
    TEXT_CONFIG_PATH = os.path.join(MODEL_DIR, "text_vectorizer_config.json")
    TEXT_VOCAB_PATH = os.path.join(MODEL_DIR, "text_vectorizer_vocab.txt")

    def __init__(self, openweather_api_key: str = None):
        """Initialize the DL service by loading model components."""
        self.openweather_api_key = openweather_api_key or os.getenv(
            "OPENWEATHERMAP_API_KEY"
        )
        if not self.openweather_api_key:
            raise ValueError(
                "OPENWEATHERMAP_API_KEY not set in environment or .env file."
            )

        self._validate_model_files()
        tf.get_logger().setLevel(logging.ERROR)

        self.model = tf.keras.models.load_model(self.MODEL_PATH)
        logger.info("DL model loaded successfully.")

        self.scaler = joblib.load(self.SCALER_PATH)
        self.label_encoder = joblib.load(self.LABEL_ENCODER_PATH)
        self.text_vectorizer = self._load_text_vectorizer()

        self._validate_model_inputs()

    def _validate_model_files(self) -> None:
        """Ensure all required model files exist."""
        required_files = [
            self.MODEL_PATH,
            self.SCALER_PATH,
            self.LABEL_ENCODER_PATH,
            self.TEXT_CONFIG_PATH,
            self.TEXT_VOCAB_PATH,
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f"Missing model files in {self.MODEL_DIR}:\n"
                + "\n".join([f"  - {os.path.basename(p)}" for p in missing])
            )

    def _load_text_vectorizer(self) -> layers.TextVectorization:
        """Load Keras TextVectorization layer from config and vocabulary."""
        try:
            with open(self.TEXT_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            with open(self.TEXT_VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f]
            vectorizer = layers.TextVectorization.from_config(config)
            vectorizer.set_vocabulary(vocab)
            logger.info("TextVectorization layer loaded.")
            return vectorizer
        except Exception as e:
            logger.error(f"Error loading TextVectorization: {e}")
            raise

    def _validate_model_inputs(self) -> None:
        """Validate model input layer names and shapes."""
        inputs = {inp.name.split(":")[0]: inp.shape for inp in self.model.inputs}
        if "numeric_input" not in inputs or "text_input" not in inputs:
            raise ValueError(
                "Model inputs must include 'numeric_input' and 'text_input'."
            )
        if inputs["numeric_input"] != EXPECTED_NUMERIC_SHAPE:
            raise ValueError(
                f"Expected numeric shape {EXPECTED_NUMERIC_SHAPE}, got {inputs['numeric_input']}"
            )
        if inputs["text_input"][1] != EXPECTED_TEXT_SHAPE[1]:
            raise ValueError(
                f"Expected text length {EXPECTED_TEXT_SHAPE[1]}, got {inputs['text_input'][1]}"
            )

    def _standardize_symptoms(self, text: str) -> str:
        """Normalize symptom text for consistent token mapping."""
        text = text.lower()
        replacements = {
            "throbbing headache": "throbbing_headache",
            "severe headache": "severe_headache",
            "pain behind the eyes": "pain_behind_the_eyes",
            "light sensitivity": "sensitivity_to_light",
            "blurred vision": "blurred_vision",
            "trouble speaking": "trouble_speaking",
            "loss of balance": "loss_of_balance",
            "shortness of breath": "shortness_of_breath",
            "chest pain": "chest_pain",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"[,:\.\-\/]", " ", text)
        text = re.sub(r"\band\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _fetch_weather(self, city: str) -> tuple:
        """Fetch live weather data using OpenWeather API."""
        default_weather = (37.0, 0.70, 18.0)
        if not city or city.lower() == "unknown":
            return default_weather
        try:
            response = requests.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": self.openweather_api_key,
                    "units": "metric",
                },
                timeout=8,
            ).json()
            if response.get("main"):
                temp = response["main"]["temp"]
                humidity = response["main"]["humidity"] / 100.0
                wind = response.get("wind", {}).get("speed", 5.0) * 3.6
                return temp, humidity, wind
        except Exception as e:
            logger.warning(f"Weather fetch failed for {city}: {e}")
        return default_weather

    async def predict(self, user_text: str) -> Dict[str, Any]:
        """Predict disease from free-text user input."""
        try:
            raw = user_text.strip()

            # safer regex matching for mypy
            age_match = re.search(r"age\s*(\d{1,3})", raw, re.I)
            if age_match:
                age = float(age_match.group(1))
            else:
                age = 30.0

            gender = 1 if re.search(r"\bmale\b", raw, re.I) else 0
            city_match = re.search(
                r"(?:from|city|living in)\s*([a-zA-Z\s]+)", raw, re.I
            )
            city = city_match.group(1).strip() if city_match else "Unknown"

            symptom_match = re.search(
                r"(?:symptom[s]*[:\-\s]*|I\s+have\s+|I\'m\s+experiencing\s+|suffering\s+from\s+)(.*)",
                raw,
                re.I,
            )
            symptoms_raw = symptom_match.group(1).strip() if symptom_match else raw
            symptoms_std = self._standardize_symptoms(symptoms_raw)
            symptom_count = len(symptoms_std.split())

            temp_c, humidity, wind_kmh = self._fetch_weather(city)

            numeric_raw = np.array(
                [[age, temp_c, humidity, wind_kmh, symptom_count]], dtype=np.float32
            )
            numeric_scaled = self.scaler.transform(numeric_raw)

            # added explicit type annotation
            binary: np.ndarray = np.zeros((1, TOTAL_BINARY_FEATURES), dtype=np.float32)
            tokens = set(symptoms_std.split())
            for i, s in enumerate(SYMPTOM_COLUMNS):
                binary[0, i] = 1.0 if s in tokens else 0.0
            binary[0, -1] = float(gender)

            X_numeric = np.hstack([numeric_scaled, binary]).astype(np.float32)
            X_text = self.text_vectorizer([symptoms_std])

            probs = self.model.predict(
                {"numeric_input": X_numeric, "text_input": X_text}, verbose=0
            )[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            disease = self.label_encoder.inverse_transform([idx])[0]

            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [
                {
                    "disease": self.label_encoder.inverse_transform([i])[0],
                    "confidence": f"{float(p):.4f}",
                }
                for i, p in zip(top5_idx, probs[top5_idx])
            ]

            return {
                "prediction": disease,
                "confidence": f"{confidence:.4f}",
                "top_5": top5,
                "debug": {
                    "age": age,
                    "gender": "male" if gender else "female",
                    "city": city,
                    "temp_c": temp_c,
                    "humidity": humidity,
                    "wind_kmh": wind_kmh,
                    "symptom_count": symptom_count,
                    "symptom_std": symptoms_std,
                },
                "status": "Success",
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e), "status": "Error"}


# SELF-TEST (manual run)
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise ValueError("Set OPENWEATHERMAP_API_KEY in .env file")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    service = DLService(api_key)

    print("\nDL Service Evaluation (Web-Ready)")
    print("=" * 60)

    cases = [
        {
            "true": "Common Cold",
            "input": "Age 26, Gender: Female from Mumbai, symptoms: fever, headache, body pain",
        },
        {
            "true": "Common Cold",
            "input": "Age 55 male, city Delhi, I have sore throat and cough",
        },
        {
            "true": "Migraine",
            "input": "Age 40 male from Delhi, symptoms: severe throbbing headache, pain behind the eyes, vomiting",
        },
        {
            "true": "Stroke",
            "input": "Age 70 female from San Francisco, I'm experiencing sudden numbness on one side, trouble speaking, blurred vision",
        },
    ]

    correct = 0
    for i, case in enumerate(cases, 1):
        result = asyncio.run(service.predict(case["input"]))
        if result["status"] == "Error":
            print(f"Test {i}: ERROR → {result['error']}")
            continue
        pred, conf = result["prediction"], result["confidence"]
        passed = pred == case["true"]
        print(f"Test {i}: {case['input'][:70]}...")
        print(
            f"  → {pred} ({conf}) → {'PASSED' if passed else f'FAILED (exp: {case['true']})'}\n"
        )
        if passed:
            correct += 1

    acc = (correct / len(cases)) * 100
    print(f"Final Accuracy: {acc:.2f}% ({correct}/{len(cases)} correct)")
    print("=" * 60)
