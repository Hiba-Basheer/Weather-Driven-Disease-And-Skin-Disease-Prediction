#!/usr/bin/env python3
"""
Auto-sync trained models from training folders → RAGentWeb/models/
Run this after training, before git commit.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  
RAGENT = ROOT / "RAGentWeb" / "models"

# DL MODEL
dl_src = ROOT / "DL_Training" / "models"
dl_dst = RAGENT / "dl"
dl_dst.mkdir(exist_ok=True)

for file in ["dl_model.keras", "scaler.pkl", "label_encoder.pkl",
             "text_vectorizer_config.json", "text_vectorizer_vocab.txt"]:
    src = dl_src / file
    dst = dl_dst / file
    if src.exists():
        shutil.copy2(src, dst)
        print(f"DL: {file}")
    else:
        print(f"DL: {file} not found")

# ML MODEL
ml_src = ROOT / "ML_Training" / "models"
ml_dst = RAGENT / "ml"
ml_dst.mkdir(exist_ok=True)

for file in ["trained_model.pkl", "label_encoder.pkl", "ml_expected_columns.pkl"]:
    src = ml_src / file
    dst = ml_dst / file
    if src.exists():
        shutil.copy2(src, dst)
        print(f"ML: {file}")
    else:
        print(f"ML: {file} not found")

# IMAGE MODEL
img_src = ROOT
img_dst = RAGENT

for file in ["models/resnet_model.h5", "data/class_labels.txt"]:
    src = img_src / file
    dst = img_dst / file.name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"IMG: {file}")
    else:
        print(f"IMG: {file} not found")

print("\nModels synced to RAGentWeb/models/")