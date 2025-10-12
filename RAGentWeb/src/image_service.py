import os
import tensorflow as tf
import numpy as np
import logging
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import preprocess_input

logger = logging.getLogger("ImageService")


class ImageClassificationService:
    """
    Image Classification Service using a pre-trained Keras CNN model (ResNet50).
    Correctly handles preprocessing and outputs top-3 predictions.
    """

    def __init__(self, model_path: str, labels_path: str):
        """
        Initializes the service by loading the Keras model and class labels.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Image classification model not found at: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Class labels file not found at: {labels_path}")

        try:
            tf.get_logger().setLevel(logging.ERROR)
            self.model = tf.keras.models.load_model(model_path)
            tf.get_logger().setLevel(logging.INFO)
            self.target_size = self.model.input_shape[1:3]
        except Exception as e:
            raise IOError(f"Failed to load Keras model: {e}")

        # Load class labels
        with open(labels_path, 'r') as f:
            self.class_labels = [line.strip() for line in f if line.strip()]

        logger.info(f"Image Classification Model loaded with {len(self.class_labels)} classes.")

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesses image bytes for model input:
        - Converts to RGB
        - Resizes to model input shape
        - Applies model-specific preprocessing (ResNet50)
        - Adds batch dimension
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.target_size)
        image_array = np.array(image, dtype=np.float32)

        # Apply ResNet50 preprocessing
        image_array = preprocess_input(image_array)

        # Add batch dimension
        return np.expand_dims(image_array, axis=0)

    def classify(self, image_bytes: bytes) -> dict:
        """
        Classifies an image and returns prediction results.
        Returns top-3 predictions with confidence.
        """
        try:
            input_tensor = self._preprocess_image(image_bytes)

            predictions = self.model.predict(input_tensor)[0]

            if len(predictions) != len(self.class_labels):
                return {
                    "error": "Mismatch between model outputs and class labels.",
                    "prediction": "N/A",
                    "status": "Error"
                }

            # Predicted class
            top_indices = np.argsort(predictions)[::-1][:3]
            top_predictions = [
                {"label": self.class_labels[i], "confidence": float(predictions[i])}
                for i in top_indices
            ]
            predicted_label = top_predictions[0]["label"]
            confidence = top_predictions[0]["confidence"]

            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "status": "Success"
            }

        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return {
                "error": f"Internal image processing error: {e}",
                "prediction": "N/A",
                "status": "Error"
            }
