import os
import logging
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import TextVectorization
import json

from preprocess import preprocess_user_input


# Logging setup

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Constants

MLFLOW_EXPERIMENT = 'DL_Module_Experiment'
LOCAL_ARTIFACTS_PATH = r"D:\brototype\week27\mlruns\212116454135138277"

def get_latest_run_id():
    """Fetches the run ID of the latest completed MLflow run."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if not experiment:
        logger.error(f"Experiment '{MLFLOW_EXPERIMENT}' not found.")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        logger.error("No runs found for the experiment.")
        return None
    return runs[0].info.run_id

def load_preprocessor_artifacts(run_id):
    """Loads the scaler, text vectorizer, and label encoder from MLflow artifacts."""
    try:
        scaler_path = os.path.join(LOCAL_ARTIFACTS_PATH, run_id, "artifacts", "scaler.pkl")
        scaler = joblib.load(scaler_path)
        logger.info("StandardScaler loaded successfully.")

        vectorizer_config_path = os.path.join(LOCAL_ARTIFACTS_PATH, run_id, "artifacts", "text_vectorizer_config.json")
        with open(vectorizer_config_path, 'r') as f:
            config = json.load(f)
        
        text_vectorizer = TextVectorization.from_config(config)
        vocab_path = os.path.join(LOCAL_ARTIFACTS_PATH, run_id, "artifacts", "text_vectorizer_vocab.txt")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        text_vectorizer.set_vocabulary(vocab)
        logger.info("TextVectorization layer loaded successfully.")

        label_encoder_path = os.path.join(LOCAL_ARTIFACTS_PATH, run_id, "artifacts", "label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        logger.info("LabelEncoder loaded successfully.")

        return scaler, text_vectorizer, label_encoder
    except Exception as e:
        logger.error(f"Failed to load preprocessor artifacts: {e}")
        return None, None, None

def predict_disease(user_full_symptoms: str):
    """
    Main function to predict disease based on user symptoms.
    Returns both predicted disease and confidence scores.
    """
    run_id = get_latest_run_id()
    if not run_id:
        return "Prediction failed: Could not find a valid MLflow run.", None
    
    scaler, text_vectorizer, label_encoder = load_preprocessor_artifacts(run_id)
    if not scaler or not text_vectorizer or not label_encoder:
        return "Prediction failed: Could not load preprocessor artifacts.", None

    # Load the model
    try:
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.tensorflow.load_model(model_uri)
        logger.info(f"Successfully loaded model from run {run_id}")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return "Prediction failed: Could not load the model.", None

    # Preprocess the user's input
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

        X_numeric_scaled, X_text_sequenced, _ = preprocess_user_input(normalized_symptoms, scaler, text_vectorizer)
    except Exception as e:
        logger.error(f"Failed to preprocess user input: {e}")
        return "Prediction failed: Error during input processing.", None

    # Make prediction
    predictions = loaded_model.predict({
        'numeric_input': X_numeric_scaled,
        'text_input': X_text_sequenced
    })
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]

    # Confidence scores for all diseases
    class_probabilities = predictions[0]
    confidence_scores = {label: float(prob) for label, prob in zip(label_encoder.classes_, class_probabilities)}

    return predicted_disease, confidence_scores

if __name__ == "__main__":
    logger.info("This script is a function library. Use real_prediction.py to run predictions.")
