"""
Standalone evaluation script for trained image classification model.
Loads saved model and evaluates on test set.
Logs metrics, classification report, and confusion matrix to MLflow.
"""

import logging
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import get_dataset, get_classes
from utils import log_metrics_to_mlflow

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Config
MODEL_PATH = "resnet_model.h5"

# MLflow setup
mlflow.set_experiment("image_classification_experiment")


def save_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """Save confusion matrix as PNG and CSV."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    # Save CSV
    df_cm.to_csv("confusion_matrix.csv")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return "confusion_matrix.csv", filename


def evaluate_model():
    """Load trained model, evaluate on test set, and log results."""
    # Load classes
    classes = get_classes()
    num_classes = len(classes)
    logging.info(f"Loaded {num_classes} classes: {classes}")

    # Load test dataset
    test_dataset = get_dataset(split="test")

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info(f"Loaded model from {MODEL_PATH}")

    # Collect predictions
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Compute metrics
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv = "classification_report.csv"
    report_df.to_csv(report_csv)

    # Save confusion matrix
    cm_csv, cm_png = save_confusion_matrix(y_true, y_pred, classes)

    # Log everything to MLflow
    with mlflow.start_run(run_name="evaluation_run"):
        mlflow.log_artifact(report_csv)
        mlflow.log_artifact(cm_csv)
        mlflow.log_artifact(cm_png)

        metrics = {
            "test_accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
        }
        log_metrics_to_mlflow(metrics)

        # Also log the model again for traceability
        mlflow.keras.log_model(model, artifact_path="model_evaluated")

    logging.info("Evaluation complete. Metrics and artifacts logged to MLflow.")


if __name__ == "__main__":
    evaluate_model()
