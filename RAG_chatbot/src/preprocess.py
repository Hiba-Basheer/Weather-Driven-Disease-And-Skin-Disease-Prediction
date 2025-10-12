"""
Unified preprocessing module for ML and DL inference.
Handles structured feature encoding and unstructured text cleaning.
"""

import re
import logging
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.symptoms import SYMPTOM_MAP

MAX_SYMPTOMS = len(SYMPTOM_MAP)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants for ML preprocessing
GENDER_MAP = {"male": 0, "female": 1}
YES_NO_MAP = {"yes": 1, "no": 0}
RASH_LOCATIONS = ["arms", "legs", "face", "torso", "none"]

# Constants for DL preprocessing
MAX_LEN = 100
VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(["placeholder"])  # Dummy fit to initialize tokenizer

def preprocess_ml_input(data: dict) -> np.ndarray:
    """
    Preprocess structured input for ML model.

    Parameters:
        data (dict): Dictionary with keys like 'age', 'gender', 'symptoms', 'city'.

    Returns:
        np.ndarray: Preprocessed feature array.
    """
    try:
        age = float(data.get("age", 0))
        gender = GENDER_MAP.get(data.get("gender", "").lower(), -1)

        raw_symptoms = data.get("symptoms", "").lower().split(",")
        symptom_vector = [0] * MAX_SYMPTOMS
        for symptom in raw_symptoms:
            symptom = symptom.strip()
            if symptom in SYMPTOM_MAP:
                symptom_vector[SYMPTOM_MAP[symptom]] = 1

        features = [age, gender] + symptom_vector
        logging.info(f"ML input preprocessed: {features}")
        return np.array(features).reshape(1, -1)
    except Exception as e:
        logging.error(f"Error preprocessing ML input: {e}")
        return None


def preprocess_dl_text(text: str) -> np.ndarray:
    """
    Preprocess unstructured text for DL model.

    Parameters:
        text (str): Raw user input.

    Returns:
        np.ndarray: Padded sequence for DL prediction.
    """
    try:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        sequences = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
        logging.info("DL text input successfully preprocessed.")
        return padded
    except Exception as e:
        logging.error(f"Text preprocessing failed: {e}")
        return None