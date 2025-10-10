import os
import sys
import logging
from dotenv import load_dotenv

# --- Logging Configuration ---
# Configures the logging system to output messages with a timestamp,
# log level, and the message itself. This helps in tracking the script's
# execution flow and debugging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Path Configuration ---
# The following code block is responsible for dynamically adding the project's
# 'src' directory to the Python system path. This allows the script to
# import modules from the 'src' folder (e.g., 'predict.py') regardless of
# where the script is executed from.
current_dir = os.path.dirname(__file__)
ml_codes_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(ml_codes_root, "src")
sys.path.insert(0, src_path)

# --- Environment Variables ---
# Loads environment variables from a .env file located at the project root.
# This is a standard practice for managing sensitive information like API keys,
# keeping them separate from the codebase.
dotenv_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".env"))
load_dotenv(dotenv_path)

# Import the main prediction function after configuring the path
from predict import predict_disease

def main():
    """
    Main function to orchestrate the disease prediction process.

    This function defines a sample user input, loads the necessary
    machine learning model and label encoder, and then calls the
    `predict_disease` function to get a prediction. It logs the
    process and prints the final result.
    """
    symptom_columns = [
        'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills', 'runny_nose',
        'pain_behind_the_eyes', 'dizziness', 'headache', 'chest_pain', 'vomiting', 'cough',
        'shivering', 'asthma_history', 'high_cholesterol', 'diabetes', 'obesity', 'hiv_aids',
        'nasal_polyps', 'asthma', 'high_blood_pressure', 'weakness', 'trouble_seeing', 'fever',
        'body_aches', 'sore_throat', 'sneezing', 'diarrhea', 'rapid_breathing', 'rapid_heart_rate',
        'pain_behind_eyes', 'swollen_glands', 'rashes', 'sinus_headache', 'facial_pain',
        'shortness_of_breath', 'skin_irritation', 'itchiness', 'throbbing_headache',
        'back_pain', 'knee_ache', 'confusion', 'fatigue', 'severe_headache', 'reduced_smell_and_taste'
    ]

    user_input = {
        'age': 20,
        'gender': 'male',
        'city': 'Kochi',
        'symptoms': ['asthma', 'sneezing', 'diarrhea', 'sore_throat']
    }

    model_path = os.path.join(ml_codes_root, "models", "trained_model.pkl")
    label_path = os.path.join(ml_codes_root, "models", "label_encoder.pkl")

    logger.info("Running prediction...")
    disease = predict_disease(user_input, model_path, label_path, symptom_columns)
    logger.info(f"Predicted Disease: {disease}")
    print(f"Predicted Disease: {disease}")

if __name__ == "__main__":
    main()