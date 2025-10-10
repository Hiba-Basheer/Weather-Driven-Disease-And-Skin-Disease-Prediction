"""
Image classification module using ResNet.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = r"D:\brototype\week27\RAG_chatbot\models\resnet_model.h5"

# Load model
img_model = tf.keras.models.load_model(MODEL_PATH)

# Correct class order used during training
CLASSES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Tinea Ringworm Candidiasis and other Fungal Infections"
]

def preprocess_image(img_path: str):
    """
    Load and preprocess image for ResNet model.

    Parameters:
        img_path (str): Path to image file.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

def predict_from_image(img_path: str):
    """
    Predict disease class from image.

    Parameters:
        img_path (str): Path to image file.

    Returns:
        tuple: (predicted class name, confidence score)
    """
    processed = preprocess_image(img_path)
    if processed is None:
        return "Error", 0.0

    preds = img_model.predict(processed)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    return CLASSES[idx], confidence