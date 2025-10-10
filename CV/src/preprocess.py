"""
Image preprocessing module for training and evaluation.
Automatically detects classes from the dataset.
"""

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DATA_DIR = r"D:\brototype\week27\CV\data\processed"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


def get_classes():
    """Detect classes automatically from processed train directory"""
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
    classes = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    classes.sort()  # Ensures consistent label order
    logging.info(f"Detected classes: {classes}")
    return classes


def get_dataset(split="train"):
    """
    Create a tf.data.Dataset from directory for a given split.
    Applies ResNet preprocessing.
    """
    classes = get_classes()
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, split),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        labels="inferred",
        label_mode="int",
        class_names=classes
    )

    dataset = dataset.map(lambda x, y: (preprocess_input(x), y))
    return dataset


def load_and_preprocess_image(img_path):
    """Load a single image and preprocess for ResNet"""
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = preprocess_input(image)
    return image
