"""
Training & Evaluation script for image classification.
Adds class weighting for imbalance and progressive fine-tuning of ResNet.
Logs all metrics and artifacts to MLflow.
"""

import logging
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from preprocess import get_dataset, get_classes
from utils import log_metrics_to_mlflow

# Config
EPOCHS_INITIAL = 3      # train only dense layers
EPOCHS_FINETUNE = 3     # fine-tune ResNet layers
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MLflow setup
mlflow.set_experiment("image_classification_experiment")


def create_resnet_model(num_classes, trainable=False):
    """Create a ResNet50-based model for classification."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = trainable

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE if not trainable else FINETUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """Save confusion matrix as PNG and CSV."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    # Save as CSV
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


def train_model():
    """Train, evaluate, and log metrics to MLflow with class weights and fine-tuning."""
    classes = get_classes()
    num_classes = len(classes)
    logging.info(f"Number of classes: {num_classes}")

    # Load datasets
    train_dataset = get_dataset(split="train")
    val_dataset = get_dataset(split="val")
    test_dataset = get_dataset(split="test")

    # Compute class weights
    logging.info("Computing class weights for imbalance handling...")
    y_train = []
    for _, labels in train_dataset:
        y_train.extend(labels.numpy())
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(y_train)
    )
    class_weights = dict(enumerate(class_weights))
    logging.info(f"Class weights: {class_weights}")

    with mlflow.start_run():
        # Step 1: Train only dense layers
        model = create_resnet_model(num_classes, trainable=False)
        history_initial = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS_INITIAL,
            class_weight=class_weights
        )

        # Step 2: Fine-tune top layers of ResNet
        logging.info("Unfreezing top layers of ResNet50 for fine-tuning...")
        model.layers[0].trainable = True
        for layer in model.layers[0].layers[:-50]:  # freeze all but top 50 layers
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=FINETUNE_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_finetune = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights
        )

        # Merge training histories
        hist_df = pd.DataFrame({
            "accuracy": history_initial.history["accuracy"] + history_finetune.history["accuracy"],
            "val_accuracy": history_initial.history["val_accuracy"] + history_finetune.history["val_accuracy"],
            "loss": history_initial.history["loss"] + history_finetune.history["loss"],
            "val_loss": history_initial.history["val_loss"] + history_finetune.history["val_loss"],
        })
        hist_csv = "training_history.csv"
        hist_df.to_csv(hist_csv, index=False)
        mlflow.log_artifact(hist_csv)

        # Evaluate on test set
        y_true, y_pred = [], []
        for images, labels in test_dataset:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        # Classification report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_csv = "classification_report.csv"
        report_df.to_csv(report_csv)
        mlflow.log_artifact(report_csv)

        # Confusion matrix
        cm_csv, cm_png = save_confusion_matrix(y_true, y_pred, classes)
        mlflow.log_artifact(cm_csv)
        mlflow.log_artifact(cm_png)

        # Log global metrics
        final_metrics = {
            "train_accuracy": hist_df["accuracy"].iloc[-1],
            "val_accuracy": hist_df["val_accuracy"].iloc[-1],
            "test_accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
        }
        log_metrics_to_mlflow(final_metrics)

        # Log per-class metrics 
        for class_name in classes:
            if class_name in report:
                safe_class_name = re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", class_name)
                if safe_class_name[0].isdigit():
                    safe_class_name = "class_" + safe_class_name

                mlflow.log_metric(f"precision_{safe_class_name}", report[class_name]["precision"])
                mlflow.log_metric(f"recall_{safe_class_name}", report[class_name]["recall"])
                mlflow.log_metric(f"f1_{safe_class_name}", report[class_name]["f1-score"])
                mlflow.log_metric(f"support_{safe_class_name}", report[class_name]["support"])

        # Save model
        model_save_path = "resnet_model.h5"
        model.save(model_save_path)
        mlflow.keras.log_model(model, artifact_path="model")
        logging.info(f"Model saved and logged to MLflow at {model_save_path}")


if __name__ == "__main__":
    train_model()
