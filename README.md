# Weather-Driven Disease and Skin Disease Prediction
 ### A Multimodel AI Platform for Proactive Health Risk Assessment

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

### 2. Multimodal Deep Learning (Text + Tabular Fusion)

✔ Purpose

Fuse free-text symptom descriptions with structured weather and demographic data for richer, context-aware predictions.

✔ Dual-Branch Architecture

Branch A – Text

Keras TextVectorization

Embedding layer

BiLSTM encoder

Branch B – Tabular

StandardScaler for continuous features

Raw symptom indicators + gender

Dense layers + Dropout

Fusion Layer

Concatenation → Dense → Dropout → Softmax classifier

✔ Additional Highlights

Synthetic text dataset generated from structured symptoms

Supports real-time weather via OpenWeatherMap API

Handles class imbalance via class weights

### 3. Deep Learning Module (Image-Based Skin Disease Detection)
This module implements image-based skin disease prediction using a Deep Learning model (ResNet50).

✔ Purpose

Detect and classify skin conditions from input images using Deep Learning.

✔ Approach

Uses ResNet50 pretrained on ImageNet

Adds custom dense layers for skin disease classification

Employs a two-phase training strategy:

Train only new dense layers while freezing ResNet50

Fine-tune the top layers of ResNet50 with a low LR

Handles class imbalance using class weights

✔ Performance

Accuracy: 96.99%

Macro F1-score: 95.78%

Reliable across 6 classes despite dataset imbalance

### 4. RAG-Based Chatbot

✔ Purpose

Provide informative, contextual disease explanations after a prediction.

✔ Stack

FAISS vector store

HuggingFace embeddings

Groq LLM 

LangChain Conversational Retrieval Chain

✔ Flow

User query →

FAISS retrieves relevant documents →

Groq LLM generates grounded response

✔ Strength

Delivers reliable, explanation-focused outputs with minimized hallucination.

### 5. Tableau Health Monitoring Dashboard

✔ Purpose

Visualize historical weather–disease patterns.

✔ Features

Disease frequency charts

Weather correlation analysis

Symptom co-occurrence exploration

Dynamic filters

✔ Data Source

Historical weather-driven disease dataset

### 6. FastAPI Web Application

✔ Purpose

Provide a unified interface for all four AI services.

✔ Architecture

Backend: FastAPI

Frontend: HTML5 + Bootstrap + Custom CSS

Services Loaded at Startup:

ML predictor

DL multimodal model

Image classifier

RAG chatbot

---
##  End-to-End System Flow

User enters symptoms, image, or query

FastAPI routes request to correct module

Model generates prediction

RAG chatbot gives medical-style explanation

UI displays results

Tableau dashboard provides historical context

---
## Project Conclusion

This work delivers a multimodal medical AI platform capable of handling:

Numerical health data

Free-text descriptions

Environmental weather attributes

Skin disease images

Informational medical queries

---
## Future Improvements

Incorporate larger, clinically verified datasets
Integrate real-time logging into Tableau dashboard
Integrate real-time weather APIs
Expand RAG knowledge base




