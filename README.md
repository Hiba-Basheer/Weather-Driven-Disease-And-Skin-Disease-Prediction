# Weather-Driven Disease and Skin Disease Prediction

## Overview
This project predicts **weather-related** and **skin-related diseases** using a combination of:
- **Image Classification (CV)**
- **Machine Learning (ML)**
- **Deep Learning (DL)**
- **Retrieval-Augmented Generation (RAG)**  

It integrates these components into a unified system with a **web-based dashboard** and **cloud deployment** for accessibility and scalability.

---

## Project Structure

```
week27/
├── CV/ # Image classification module
│ └── src/ # Preprocessing, training, prediction
├── DL/ # Deep learning module
│ ├── DL_module/
│ │ └── dl_codes/ # Model training and prediction scripts
│ ├── artifacts/ # Temporary model outputs (ignored by git)
│ └── mlflow_artifacts/ # MLflow tracking outputs (ignored by git)
├── ML/ # Machine learning module
│ └── ML_module/ml_codes/
├── RAG_chatbot/ # Retrieval-augmented chatbot logic
│ └── src/
├── RAGentWeb/ # Web app for model access and dashboard
│ ├── data/ # Weather and skin datasets
│ └── src/ # API backend and web services
├── .gitignore
└── README.md

```
---

## Modules Description

### 1. CV (Image classification)
Handles preprocessing and training of image-based models for disease detection.

### 2. DL (Deep Learning)
Deep learning pipelines for disease prediction.
Includes MLflow for experiment tracking.

### 3. ML (Machine Learning)
Classic machine learning pipelines and research experiments on disease prediction. Includes Jupyter notebooks for experimentation.

### 4. RAG_chatbot
Retrieval-based chatbot module to provide users with disease and weather-related guidance.

### 5. RAGentWeb
Web interface for accessing predictions, chatbot, and visual dashboards.

---


### Future Improvements
 
Integrate real-time weather APIs
Expand RAG knowledge base

