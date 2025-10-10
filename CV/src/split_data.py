"""
Automatically split dataset into train, validation, and test sets.
Handles any number of classes dynamically.
"""

import os
import shutil
import random
import logging

# Paths
RAW_DATA_DIR = r"D:\brototype\week27\CV\data\raw"
PROCESSED_DATA_DIR = r"D:\brototype\week27\CV\data\processed"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_classes():
    """Detect all class folders in RAW_DATA_DIR"""
    classes = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
    classes.sort()  # Important for consistent label ordering
    return classes


def split_dataset():
    """Split images into train, validation, and test directories."""
    classes = get_classes()
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(RAW_DATA_DIR, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(TRAIN_RATIO * n_total)
        n_val = int(VAL_RATIO * n_total)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(PROCESSED_DATA_DIR, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(cls_path, img), os.path.join(split_dir, img))

        logging.info(f"Split done for class '{cls}' with {n_total} images.")


if __name__ == "__main__":
    split_dataset()
    classes = get_classes()
    logging.info(f"Classes detected: {classes}")
