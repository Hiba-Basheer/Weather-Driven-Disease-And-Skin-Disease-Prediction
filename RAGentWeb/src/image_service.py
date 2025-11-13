"""
image_service.py
Image Classification Service using a pre-trained CNN model (e.g., ResNet50).

This module provides an 'ImageClassificationService' class that loads a trained
Keras image classification model and corresponding label file to predict disease
or category labels from input images. It correctly handles preprocessing,
prediction, and outputs top-3 results with confidence scores.
"""

import logging
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger("ImageService")


class ImageClassificationService:
    """
    Image Classification Service using a pre-trained Keras CNN model (ResNet50).

    This service loads a saved model and label file, preprocesses input images,
    and performs inference to return top-3 predictions with associated confidence scores.
    """

    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the image classification service.

        Args:
            model_path (str): Path to the trained Keras model (.h5 or .keras).
            labels_path (str): Path to the class labels text file.

        Raises:
            FileNotFoundError: If the model or labels file cannot be found.
            IOError: If the Keras model fails to load.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Image classification model not found at: {model_path}"
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Class labels file not found at: {labels_path}")

        try:
            tf.get_logger().setLevel(logging.ERROR)
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.target_size = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])
            tf.get_logger().setLevel(logging.INFO)
        except Exception as e:
            raise IOError(f"Failed to load TFLite model: {e}")

        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        logger.info(
            f"Image Classification Model loaded with {len(self.labels)} classes."
        )

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image bytes for model input.

        Steps:
            1. Converts image bytes into a PIL RGB image.
            2. Resizes image to match model input shape.
            3. Converts image to NumPy array and applies model-specific preprocessing (ResNet50).
            4. Adds a batch dimension for inference.

        Args:
            image_bytes (bytes): Raw image bytes.

        Returns:
            np.ndarray: Preprocessed image tensor ready for model input.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.target_size)
        image_array = np.array(image, dtype=np.float32)

        image_array = image_array / 127.5 - 1.0

        return np.expand_dims(image_array, axis=0)

    def classify(self, image_bytes: bytes) -> dict:
        """
        Classify an input image and return the top predictions.

        Args:
            image_bytes (bytes): Raw image bytes to classify.

        Returns:
            dict: A structured result containing:
                - "prediction" (str): Most likely label.
                - "confidence" (float): Confidence of top prediction.
                - "top_predictions" (list): Top-3 predictions with confidences.
                - "status" (str): "Success" or "Error".
                - "error" (optional, str): Error message if an exception occurred.
        """
        try:
            input_tensor = self._preprocess_image(image_bytes)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            if len(predictions) != len(self.labels):
                return {
                    "error": "Mismatch between model outputs and class labels.",
                    "prediction": "N/A",
                    "status": "Error",
                }

            top_indices = np.argsort(predictions)[::-1][:3]
            top_predictions = [
                {"label": self.labels[i], "confidence": float(predictions[i])}
                for i in top_indices
            ]
            predicted_label = top_predictions[0]["label"]
            confidence = top_predictions[0]["confidence"]

            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "status": "Success",
            }

        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return {
                "error": f"Internal image processing error: {e}",
                "prediction": "N/A",
                "status": "Error",
            }


if __name__ == "__main__":
    """
    Evaluation script for ImageClassificationService.

    Evaluates the classification model on a small validation set and reports:
      Top-1 accuracy (correct first prediction)
      Top-3 accuracy (correct within top 3 predictions)
      Average inference latency

    Example use:
        python image_service.py
    """
    import time
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    MODEL_PATH = r"D:\brototype\week27\RAGentWeb\models\resnet_model_optimized.tflite"
    LABELS_PATH = r"D:\brototype\week27\RAGentWeb\models\labels.txt"

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(LABELS_PATH).exists():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    GROUND_TRUTH = {
        "1. Eczema 1677": "Eczema",
        "2. Melanoma 15.75k": "Melanoma",
        "3. Atopic Dermatitis - 1.25k": "Atopic Dermatitis",
    }

    IMAGE_PATHS = [
        r"D:\brototype\week27\CV\data\processed\val\1. Eczema 1677\0_15.jpg",
        r"D:\brototype\week27\CV\data\processed\val\2. Melanoma 15.75k\ISIC_6653780.jpg",
        r"D:\brototype\week27\CV\data\processed\val\3. Atopic Dermatitis - 1.25k\0_17.jpg",
    ]

    service = ImageClassificationService(
        model_path=str(MODEL_PATH), labels_path=str(LABELS_PATH)
    )

    correct_top1 = 0
    correct_top3 = 0
    total = len(IMAGE_PATHS)
    times = []

    print("\nImage Classification Service Evaluation")
    print("=" * 80)

    for idx, img_path in enumerate(IMAGE_PATHS, 1):
        path = Path(img_path)
        if not path.exists():
            print(f"Warning: Image {idx} not found: {img_path}")
            continue

        folder_name = path.parent.name
        expected = GROUND_TRUTH.get(
            folder_name.split()[0] + ". " + " ".join(folder_name.split()[1:3]),
            "Unknown",
        )

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        start = time.time()
        result = service.classify(image_bytes)
        elapsed = time.time() - start
        times.append(elapsed)

        if result["status"] != "Success":
            print(f"Test {idx} | ERROR: {result.get('error')}")
            continue

        pred_label = result["prediction"]
        confidence = result["confidence"]
        top3 = result["top_predictions"]

        top1_match = pred_label == expected
        top3_match = any(p["label"] == expected for p in top3)

        if top1_match:
            correct_top1 += 1
        if top3_match:
            correct_top3 += 1

        print(f"\nTest {idx} | Image: {path.name}")
        print(f"Ground Truth: {expected}")
        print(f"Prediction : {pred_label} ({confidence:.4f})")
        print("Top-3:")
        for item in top3:
            marker = "Correct" if item["label"] == expected else ""
            print(f"  {item['label']:25} {item['confidence']:.4f} {marker}")
        print(
            f"Time: {elapsed:.3f}s | Top-1: {'PASS' if top1_match else 'FAIL'} | Top-3: {'PASS' if top3_match else 'FAIL'}"
        )
        print("-" * 80)

    acc_top1 = correct_top1 / total * 100
    acc_top3 = correct_top3 / total * 100
    avg_time = sum(times) / len(times) if times else 0.0

    print("\nFINAL REPORT")
    print(f"Top-1 Accuracy : {correct_top1}/{total} ({acc_top1:.1f}%)")
    print(f"Top-3 Accuracy : {correct_top3}/{total} ({acc_top3:.1f}%)")
    print(f"Avg Inference  : {avg_time:.3f}s")
    print("=" * 80)
    