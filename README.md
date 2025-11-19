# Weather-Driven Disease and Skin Disease Prediction
 ## A Multimodel AI Platform for Proactive Health Risk Assessment

## Overview
This project delivers an integrated AI-powered health prediction system capable of identifying both weather-driven diseases and skin-related conditions using a combination of:
Machine Learning (ML) – Tabular disease prediction

Multimodal Deep Learning (DL) – Text + weather + symptom fusion

Deep Learning – Image-based skin disease detection

RAG Chatbot – Retrieval-augmented guidance

Web Dashboard – Historical disease analytics using Tableau

FastAPI Web App – Unified system deployment

The platform processes structured health data, free-text symptom descriptions, environmental weather conditions, and medical images to perform robust and scalable predictions.

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

### 1. Machine Learning Module (Weather-Driven Disease Prediction)
✔ Purpose

Predict diseases using structured data: symptoms, age, gender, and weather (temperature, humidity, wind speed).

✔ Key Features

Models trained: Random Forest and XGBoost

Random Forest selected

Accuracy: 0.9856

Recall: 0.9864

Serialized via .pkl for real-time prediction

Tracked with MLflow (runs, metrics, parameters)

✔ Key Challenges

Limited availability of perfectly aligned symptom–weather datasets

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


