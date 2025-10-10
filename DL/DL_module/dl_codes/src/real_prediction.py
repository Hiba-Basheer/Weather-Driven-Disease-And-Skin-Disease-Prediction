import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import joblib
import json
import mlflow
from tabulate import tabulate

from preprocess import preprocess_user_input, SYMPTOM_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants
MLFLOW_ARTIFACTS_PATH = "mlflow_artifacts"
os.makedirs(MLFLOW_ARTIFACTS_PATH, exist_ok=True)

def load_preprocessors_from_mlflow(run_id):
    """Loads preprocessors from a specific MLflow run's artifacts."""
    try:
        # Download artifacts to a local directory
        client = mlflow.tracking.MlflowClient()
        artifact_path = os.path.join(MLFLOW_ARTIFACTS_PATH, run_id)
        client.download_artifacts(run_id=run_id, path="", dst_path=artifact_path)
        
        logger.info(f"Downloaded artifacts from MLflow run {run_id} to {artifact_path}")
        
        # Load scaler
        scaler_path = os.path.join(artifact_path, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Load label encoder
        label_encoder_path = os.path.join(artifact_path, "label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        
        # Load TextVectorization layer
        vectorizer_config_path = os.path.join(artifact_path, "text_vectorizer_config.json")
        with open(vectorizer_config_path, 'r') as f:
            vectorizer_config = json.load(f)
            
        vocab_path = os.path.join(artifact_path, "text_vectorizer_vocab.txt")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
            
        vectorizer = layers.TextVectorization.from_config(vectorizer_config)
        vectorizer.set_vocabulary(vocab)
        
        logger.info("Successfully loaded preprocessors from MLflow.")
        return scaler, label_encoder, vectorizer
    except Exception as e:
        logger.error(f"Error loading preprocessors from MLflow run {run_id}: {e}")
        return None, None, None

def predict_disease(user_input: str):
    """
    Predicts the disease based on user symptoms using the latest trained model from MLflow.
    """
    try:
        # Find the latest MLflow run
        client = mlflow.tracking.MlflowClient()
        latest_run = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("DL_Module_Experiment").experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )[0]
        run_id = latest_run.info.run_id
        
        logger.info(f"Found latest MLflow run with ID: {run_id}")
        
        # Load the model and preprocessors
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.tensorflow.load_model(model_uri)
        scaler, label_encoder, text_vectorizer = load_preprocessors_from_mlflow(run_id)

        if not all([model, scaler, label_encoder, text_vectorizer]):
            return "Error: Could not load required model or preprocessors.", None
            
        # Preprocess the user's input
        X_numeric, X_text, _ = preprocess_user_input(user_input, scaler, text_vectorizer)
        
        # Make a prediction
        prediction = model.predict([X_numeric, X_text])
        predicted_class_index = np.argmax(prediction)
        predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get confidence scores for all diseases
        class_probabilities = prediction[0]
        confidence_scores = {
            label: float(prob) for label, prob in zip(label_encoder.classes_, class_probabilities)
        }
        
        return predicted_disease, confidence_scores
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return f"Prediction Error: {e}", None

def display_results_as_table(confidence_scores):
    """
    Formats and displays prediction results in a clean table.
    """
    headers = ["Disease", "Confidence Score (%)"]
    
    # Sort the dictionary by confidence score in descending order
    sorted_scores = sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Format for the table
    table_data = [[disease, f"{score * 100:.2f}"] for disease, score in sorted_scores]
    
    # Print the table
    print("\n--- Prediction Confidence Scores ---")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print("------------------------------------\n")


if __name__ == "__main__":
    # Example user input for testing
    user_symptoms = 'I am a 29-year-old female living in San Francisco. I have been experiencing joint pain, stiffness, swelling in the joints, difficulty moving, fatigue, and sometimes mild fever. I am concerned these symptoms might indicate arthritis and would like guidance on the next steps for testing and care.'
    
    predicted_disease, confidence_scores = predict_disease(user_symptoms)
    
    if predicted_disease and confidence_scores:
        logger.info(f"Predicted disease: {predicted_disease}")
        display_results_as_table(confidence_scores)
        print("\n**Disclaimer:** This AI prediction is for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.")
    else:
        logger.info(f"Prediction failed: {predicted_disease}")
