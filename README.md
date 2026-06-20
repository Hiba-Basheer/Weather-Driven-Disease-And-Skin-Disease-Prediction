# 🌦️ Weather-Driven Disease and Skin Disease Prediction

## A Multimodal AI Platform for Proactive Health Risk Assessment

---

## Overview

Weather-Driven Disease and Skin Disease Prediction is an end-to-end AI platform that combines Machine Learning, Deep Learning, Computer Vision, Retrieval-Augmented Generation (RAG), and FastAPI to provide intelligent health risk assessment.

The platform predicts diseases from structured health information, free-text symptom descriptions, environmental weather conditions, and skin images. It also includes an AI-powered chatbot that provides contextual explanations based on retrieved medical knowledge instead of relying solely on a language model's internal knowledge.

The project demonstrates the complete AI application lifecycle—from data preprocessing and model training to deployment, API integration, and user interaction.

---

# 🚀 Project Highlights

* End-to-end AI healthcare platform
* Machine Learning disease prediction
* Multimodal Deep Learning (Text + Weather + Symptoms)
* Image-based skin disease detection using ResNet50
* Retrieval-Augmented Generation (RAG) chatbot
* FastAPI backend serving multiple AI models
* Dockerized deployment
* MLflow experiment tracking
* Tableau analytics dashboard
* Modular and scalable architecture

---

# 💡 Why This Project?

Most healthcare prediction systems focus on only one type of input, such as structured patient records or medical images.

This project explores a multimodal approach by combining:

* Structured patient information
* Weather conditions
* Free-text symptom descriptions
* Skin disease images
* Context-aware medical explanations using Retrieval-Augmented Generation (RAG)

Instead of fine-tuning a Large Language Model, I chose a Retrieval-Augmented Generation (RAG) architecture. This allows the chatbot to retrieve relevant medical information during inference, making it easier to update the knowledge base without retraining the model while producing more grounded and reliable responses.

---

# 🛠 Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* BiLSTM
* ResNet50

### LLM & RAG

* LangChain
* FAISS
* HuggingFace Embeddings
* Groq LLM

### Backend

* FastAPI

### Deployment

* Docker
* Google Cloud Platform (GCP)

### Visualization

* Tableau

### Experiment Tracking

* MLflow

---

# 🏗 System Architecture

```text
                    User
                      │
                      ▼
                  FastAPI API
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 ML Prediction   DL Prediction   Skin Disease Detection
      │               │                │
      └───────────────┼────────────────┘
                      ▼
               RAG Chatbot
                      │
               LangChain + FAISS
                      │
                   Groq LLM
                      │
                      ▼
              Health Guidance
```

---

# 📂 Project Structure

```text
week27/
├── CV/
│   └── src/
├── DL/
│   ├── DL_module/
│   ├── artifacts/
│   └── mlflow_artifacts/
├── ML/
│   └── ML_module/
├── RAG_chatbot/
│   └── src/
├── RAGentWeb/
│   ├── data/
│   └── src/
├── .gitignore
└── README.md
```

---

# 📌 Modules

## 1️⃣ Machine Learning Module

### Purpose

Predict weather-driven diseases using structured patient information.

### Inputs

* Symptoms
* Age
* Gender
* Temperature
* Humidity
* Wind Speed

### Model

* Random Forest
* XGBoost

Random Forest was selected as the final model.

### Performance

* Accuracy: **98.56%**
* Recall: **98.64%**

### Features

* MLflow experiment tracking
* Pickle model serialization
* Real-time prediction

---

## 2️⃣ Multimodal Deep Learning Module

### Purpose

Combine structured health information with natural language symptom descriptions.

### Architecture

**Text Branch**

* TextVectorization
* Embedding Layer
* BiLSTM

**Tabular Branch**

* Weather features
* Demographics
* Symptom indicators
* Dense Layers

Both branches are fused into a shared classifier for prediction.

### Features

* Synthetic text generation
* OpenWeatherMap integration
* Class imbalance handling

---

## 3️⃣ Skin Disease Detection

### Purpose

Detect skin diseases from uploaded images.

### Model

* ResNet50 (ImageNet pretrained)

### Training Strategy

* Freeze pretrained layers
* Train custom classifier
* Fine-tune upper ResNet layers

### Performance

* Accuracy: **96.99%**
* Macro F1 Score: **95.78%**

---

## 4️⃣ Retrieval-Augmented Generation (RAG)

### Purpose

Provide contextual explanations after predictions.

### Pipeline

User Query

↓

FAISS Retrieval

↓

Relevant Medical Documents

↓

Groq LLM

↓

Grounded Response

### Technologies

* LangChain
* FAISS
* HuggingFace Embeddings
* Groq

### Benefits

* Reduces hallucinations
* Easy knowledge base updates
* Grounded responses

---

## 5️⃣ Tableau Dashboard

Interactive dashboard for visualizing:

* Disease frequency
* Weather correlations
* Symptom patterns
* Historical trends

---

## 6️⃣ FastAPI Web Application

Acts as the central API that integrates:

* Machine Learning prediction
* Multimodal Deep Learning
* Skin Disease Detection
* RAG Chatbot

Provides a unified interface for all AI services.

---

# 🔄 End-to-End Workflow

```text
User Input
     │
     ▼
FastAPI Backend
     │
     ├── ML Disease Prediction
     ├── Multimodal DL Prediction
     ├── Skin Disease Detection
     └── RAG Chatbot
               │
               ▼
Prediction + Contextual Explanation
               │
               ▼
User Interface
               │
               ▼
Tableau Dashboard
```

---

# ⚙ Installation

```bash
git clone <repository-url>

cd Weather-Driven-Disease-And-Skin-Disease-Prediction

pip install -r requirements.txt

uvicorn app:app --reload
```

---

# 🚀 Future Roadmap

* Integrate larger clinical datasets
* Expand the RAG knowledge base
* Improve explainability with SHAP
* Authentication and user management

---

# 📖 Key Takeaways

This project demonstrates the design and deployment of a complete AI application that combines traditional Machine Learning, Deep Learning, Computer Vision, Retrieval-Augmented Generation, and scalable backend development.

Beyond model development, it focuses on system integration, experiment tracking, API design, deployment, and building production-oriented AI workflows.
