import os
import joblib
import numpy as np
import logging
import json
import pandas as pd
from .weather_fetcher import fetch_and_log_weather_data 

logger = logging.getLogger("MLService")

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

class MLService:
    """
    Service class for Machine Learning (ML) prediction.
    Loads a pre-trained scikit-learn model and uses structured user data
    along with real-time weather data for prediction.
    """
    FEATURE_NAMES_FILE = "feature_names.json" 

    def __init__(self, model_path: str, api_key: str):
        """
        Initializes the MLService by loading the model, feature names, and API key.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ML model not found at: {model_path}")
        
        # Determine the directory where the model file resides
        model_dir = os.path.dirname(model_path)
        feature_names_path = os.path.join(model_dir, self.FEATURE_NAMES_FILE)

        if not os.path.exists(feature_names_path):
             logger.warning(f"Feature names file NOT FOUND at: {feature_names_path}. Prediction will likely fail.")
             self.expected_features = None 
        else:
             with open(feature_names_path, 'r') as f:
                 self.expected_features = json.load(f)
             logger.info(f"Loaded {len(self.expected_features)} feature names for ML prediction.")
        
        self.model = joblib.load(model_path)
        self.api_key = api_key
        self.weather_features = ['temp', 'humidity', 'wind_speed'] 
        logger.info(f"ML Model loaded from {model_path}")

    async def predict(self, age: int, gender: str, city: str, symptoms: str) -> dict:
        """
        Fetches current weather data for the city and performs a prediction using structured inputs.
        
        Args:
            age: Age of the person.
            gender: Gender of the person (Male/Female).
            city: City name for weather data.
            symptoms: Additional symptoms (free text).

        Returns:
            A dictionary containing the prediction result (disease name), confidence, weather data, and user info.
        """
        
        # 1. Fetch Weather Data 
        weather_data = await fetch_and_log_weather_data(city, self.api_key, symptoms, service_type='ML')

        if 'error' in weather_data:
            return {
                "error": weather_data['error'],
                "prediction": "N/A",
                "status": "Failed to retrieve weather data."
            }

        # 2. Combine and Align Features 
        try:
            # Prepare the raw input dictionary for DataFrame creation
            raw_input = {
                'age': age,
                'gender': gender.lower(), 
                'city': city.lower(),
                'symptoms': symptoms.lower()
            }
            
            # Add weather features
            for name in self.weather_features:
                 raw_input[name] = weather_data.get(name, 0)

            # Create DataFrame
            input_df = pd.DataFrame([raw_input])

            # Apply One-Hot Encoding (OHE) for categorical variables
            categorical_cols = ['gender', 'city', 'symptoms']
            processed_df = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)

            # 3. Align the DataFrame columns with the expected features
            if self.expected_features is None:
                raise ValueError("Expected feature names list is missing. Cannot align input data.")

            # Reindex to ensure all 53 features are present, filling missing (not-present symptoms/cities) with 0
            final_input_df = processed_df.reindex(columns=self.expected_features, fill_value=0)
            
            # Check if feature count is correct before prediction
            if final_input_df.shape[1] != len(self.expected_features):
                raise ValueError(
                    f"Feature count mismatch after processing. Expected {len(self.expected_features)} features, but got {final_input_df.shape[1]}. "
                )

            # Convert to numpy array for prediction
            input_data = final_input_df.values
            
            # 4. Perform Prediction
            prediction_proba = self.model.predict_proba(input_data)[0]
            predicted_class_index = self.model.classes_[np.argmax(prediction_proba)]
            
            if predicted_class_index < len(DISEASE_LABELS):
                predicted_label = DISEASE_LABELS[predicted_class_index]
            else:
                predicted_label = f"Unknown Class {predicted_class_index}"
            
            # 5. Format Results
            return {
                "prediction": predicted_label, # Returns the disease name
                "confidence": float(np.max(prediction_proba)),
                "raw_weather_data": {k: float(v) for k, v in weather_data.items() if isinstance(v, (int, float))},
                "status": "Success",
                "user_info": {
                    "age": age,
                    "gender": gender,
                    "symptoms": symptoms,
                    "city": city
                }
            }

        except Exception as e:
            logger.error(f"ML prediction error for city {city}: {e}")
            return {
                "error": f"Internal prediction processing error: {e}",
                "prediction": "N/A",
                "status": "Error"
            }
