"""
Utility functions for image classification.
"""

import pandas as pd
import logging
import os
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_metrics(metrics_dict, filename="metrics.csv"):
    """
    Save metrics dictionary to a CSV file.
    """
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filename, index=False)
    logging.info(f"Metrics saved to {filename}")
    return filename


def log_metrics_to_mlflow(metrics_dict, artifact_path="metrics"):
    """
    Log metrics and CSV file to MLflow.
    """
    metrics_file = save_metrics(metrics_dict)
    mlflow.log_artifact(metrics_file, artifact_path)
    logging.info("Metrics logged to MLflow")
