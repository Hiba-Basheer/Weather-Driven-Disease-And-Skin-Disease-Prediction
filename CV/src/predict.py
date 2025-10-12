"""
Prediction script using the trained model.
"""

import tensorflow as tf
from preprocess import load_and_preprocess_image
from config import CLASSES
import numpy as np
import logging
import mlflow
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def predict_image(img_path, model_path=r"D:\brototype\week27\CV\resnet_model.h5"):
    """
    Predict class of a single image and log results to MLflow.

    Args:
        img_path (str): Path to the image.
        model_path (str): Path to the trained model.

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess
    image = load_and_preprocess_image(img_path)
    image = np.expand_dims(image, axis=0)

    # Predict
    preds = model.predict(image)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = CLASSES[class_idx]
    confidence = preds[0][class_idx]

    # Caution if confidence is low
    caution_message = None
    if confidence < 0.80:
        caution_message = (
            "Prediction confidence is below 80%. "
            "Please consult a medical professional for confirmation."
        )
        logging.warning(caution_message)

    logging.info(f"Predicted: {class_name} ({confidence:.2f})")

    # Save prediction to CSV
    results_df = pd.DataFrame([{
        "image": os.path.basename(img_path),
        "predicted_class": class_name,
        "confidence": confidence,
        "caution": caution_message if caution_message else ""
    }])
    results_path = "prediction_results.csv"
    results_df.to_csv(results_path, index=False)

    # Log as MLflow artifact
    with mlflow.start_run(run_name="prediction") as run:
        mlflow.log_artifact(results_path, artifact_path="predictions")

    return class_name, confidence, caution_message


if __name__ == "__main__":
    test_image = r"D:\brototype\week27\CV\data\processed\val\1. Eczema 1677\0_14.jpg"
    predicted_class, confidence, caution = predict_image(test_image)

    print(f"Prediction: {predicted_class} ({confidence:.2f})")
    if caution:
        print(caution)
