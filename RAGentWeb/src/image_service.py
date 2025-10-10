import os
import tensorflow as tf
import numpy as np
import logging
from PIL import Image
from io import BytesIO

logger = logging.getLogger("ImageService")

class ImageClassificationService:
    """
    Service class for Image Classification using a pre-trained Keras model (ResNet50).
    """
    def __init__(self, model_path: str, labels_path: str):
        """
        Initializes the ImageClassificationService by loading the model and class labels.

        Args:
            model_path: Path to the Keras model file.
            labels_path: Path to a text file containing class labels, one per line.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Image classification model not found at: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Class labels file not found at: {labels_path}")
        
        # Load the Keras model
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
        """Loads, resizes, and normalizes the image data."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.target_size)
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0 
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)

    def classify(self, image_bytes: bytes) -> dict:
        """
        Processes an image and returns the classification result.

        Args:
            image_bytes: Raw bytes of the uploaded image file.

        Returns:
            A dictionary containing the classification result.
        """
        try:
            # 1. Preprocess the image
            input_tensor = self._preprocess_image(image_bytes)
            
            # 2. Perform Prediction
            predictions = self.model.predict(input_tensor)[0]
            
            predicted_index = np.argmax(predictions)
            confidence = float(predictions[predicted_index])
            predicted_label = self.class_labels[predicted_index]
            
            # 3. Format Results (Top 3 predictions)
            top_indices = np.argsort(predictions)[::-1][:3]
            top_predictions = [
                {"label": self.class_labels[i], "confidence": float(predictions[i])}
                for i in top_indices
            ]
            
            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "status": "Success"
            }
            
        except IndexError:
             # Handle case where predicted index is out of bounds for class labels
             return {"error": "Classification failed: Model output mismatch with class labels.", "prediction": "N/A", "status": "Error"}
        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return {"error": f"Internal image processing error: {e}", "prediction": "N/A", "status": "Error"}
