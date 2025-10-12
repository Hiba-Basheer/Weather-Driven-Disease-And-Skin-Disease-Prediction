# Weather-Driven Disease and Skin Disease Prediction

## Overview
This project predicts weather-related and skin-related diseases using a combination of **image classification**, **machine learning**, **deep learning**, and **retrieval-augmented generation (RAG)** techniques. The project is divided into multiple modules, each with its own functionality, allowing for modular development and easier maintenance.

---

## Project Structure

```
week27/
├── CV/ # Image classification module
│ └── src/ # CV source code (preprocessing, training, prediction)
├── DL/ # Deep Learning module
│ ├── DL_module/
│ │ └── dl_codes/ # Training and prediction scripts
│ ├── artifacts/ # Generated artifacts (ignored by git)
│ └── mlflow_artifacts/ # MLflow experiment outputs (ignored by git)
├── ML/ # Machine Learning module
│ └── ML_module/
│ └── ml_codes/ # Training, prediction, and research notebooks
├── RAG_chatbot/ # Retrieval-augmented chatbot module
│ └── src/ # Chatbot logic and data handling
├── RAGentWeb/ # Web interface for ML, DL, Image classification, RAG, and Dashboard
│ ├── data/ # Weather and skin disease datasets
│ └── src/ # Web backend and services
├── .gitignore # Git ignore file (envs, artifacts, etc.)
└── README.md # Project overview


```
---

## Modules Description

### 1. CV (Image classification)
Handles preprocessing and training of image-based models for disease detection.

### 2. DL (Deep Learning)
Contains deep learning pipelines for disease prediction. Uses MLflow for experiment tracking.  
- `DL_module/dl_codes/` → core scripts for dataset creation, training, and prediction  
- `artifacts/` → temporary model outputs (ignored in git)  
- `mlflow_artifacts/` → MLflow experiment results (ignored in git)

### 3. ML (Machine Learning)
Classic machine learning pipelines and research experiments on disease prediction. Includes Jupyter notebooks for experimentation.

### 4. RAG_chatbot
Retrieval-based chatbot module to provide users with disease and weather-related guidance. Includes preprocessing and pipeline scripts.

### 5. RAGentWeb
Web interface for accessing the RAG system and visualizing data.  
- `data/` → datasets (weather, skin diseases)  
- `src/` → API and service scripts for the web app  

---

## Setup Instructions



