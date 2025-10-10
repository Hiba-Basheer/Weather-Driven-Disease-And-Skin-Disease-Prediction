"""
Configuration file for image classification.
"""

import os

# Paths
RAW_DATA_DIR = r"D:\brototype\week27\CV\data\raw"
PROCESSED_DATA_DIR = r"D:\brototype\week27\CV\data\processed"
MODEL_DIR = r"D:\brototype\week27\CV\models"
REPORT_DIR = r"D:\brototype\week27\CV\reports"

# Image properties
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Training parameters
EPOCHS = 3
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

# MLflow
MLFLOW_EXPERIMENT_NAME = "image_classification_experiment"

# Class mapping 
CLASSES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Melanocytic Nevi",
    "Tinea Ringworm Candidiasis"
]
