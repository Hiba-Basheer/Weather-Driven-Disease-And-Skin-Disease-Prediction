"""
main.py

FastAPI backend for the Health AI Predictor & RAGent Web API.

This application provides unified REST endpoints for:
    • ML-based disease prediction (structured input + weather)
    • DL-based disease prediction (text input)
    • Image-based skin disease classification
    • RAG-based medical question answering

All AI services are initialized on startup and exposed via FastAPI routes.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .dl_service import DLService
from .image_service import ImageClassificationService
from .ml_service import MLService
from .rag_service import RAGService


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] — %(message)s",
)
logger = logging.getLogger("HealthAI_API")


# Environment Setup

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
if not OPENWEATHER_API_KEY:
    logger.warning("Environment variable 'OPENWEATHERMAP_API_KEY' not found.")

BASE_DIR = Path(__file__).resolve().parent.parent


# Global Service Instances

ml_service: Optional[MLService] = None
dl_service: Optional[DLService] = None
image_service: Optional[ImageClassificationService] = None
rag_service: Optional[RAGService] = None


# FastAPI Initialization

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context for startup and shutdown.

    Initializes all AI service components (ML, DL, Image, and RAG)
    when the application starts, and gracefully shuts them down when
    the application terminates.
    """
    global ml_service, dl_service, image_service, rag_service

    logger.info("Application startup — initializing AI services...")
    start_time = time.time()

    # ML Service
    try:
        ml_model_dir = BASE_DIR / "models" / "ml"
        if not ml_model_dir.exists():
            raise FileNotFoundError(f"ML model directory not found: {ml_model_dir}")

        ml_service = MLService(str(ml_model_dir), OPENWEATHER_API_KEY)
        logger.info("ML Service initialized successfully.")
    except Exception as exc:
        logger.exception(f"Failed to initialize ML Service: {exc}")

    # DL Service
    try:
        dl_service = DLService(OPENWEATHER_API_KEY)
        logger.info("DL Service initialized successfully.")
    except Exception as exc:
        logger.exception(f"Failed to initialize DL Service: {exc}")

    # Image Classification Service
    try:
        model_path = BASE_DIR / "models" / "resnet_model.h5"
        labels_path = BASE_DIR / "models" / "class_labels.txt"
        image_service = ImageClassificationService(str(model_path), str(labels_path))
        logger.info("Image Classification Service initialized successfully.")
    except Exception as exc:
        logger.exception(f"Failed to initialize Image Classification Service: {exc}")

    # RAG Service
    try:
        faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
        rag_service = RAGService(str(faiss_path))
        logger.info(f"RAG Service initialized successfully from: {faiss_path}")
    except Exception as exc:
        logger.exception(f"Failed to initialize RAG Service: {exc}")

    elapsed = time.time() - start_time
    logger.info(f"All services loaded in {elapsed:.2f} seconds.")

    # Small delay for deployment environments (Cloud Run)
    time.sleep(3)

    yield

    logger.info("Shutting down services...")


# Application Setup

app = FastAPI(
    title="Health AI Predictor & RAGent Web API",
    description="Unified API for ML, DL, Image Classification, and RAG-based Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# Health Check Endpoint

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Returns the current health status of the application and its services.
    """
    return {
        "status": "healthy",
        "services": {
            "ml": ml_service is not None,
            "dl": dl_service is not None,
            "image": image_service is not None,
            "rag": rag_service is not None,
        },
        "message": "RAGentWeb API is operational.",
    }


# Pydantic Models for Request Payloads

class MLPredictionRequest(BaseModel):
    """Schema for ML prediction requests."""
    age: int
    gender: str
    city: str
    symptoms: str


class DLPredictionRequest(BaseModel):
    """Schema for DL prediction requests."""
    note: str


class RAGQueryRequest(BaseModel):
    """Schema for RAG question-answering requests."""
    query: str


# Routes

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Serves the frontend index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict_ml")
async def predict_ml(payload: MLPredictionRequest):
    """
    Predicts disease likelihood using a traditional ML model with weather context.
    """
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML Service not available.")

    try:
        result = await ml_service.predict({
            "age": payload.age,
            "gender": payload.gender,
            "city": payload.city,
            "symptoms": payload.symptoms,
        })
        return result
    except Exception as exc:
        logger.exception(f"ML Prediction Error: {exc}")
        raise HTTPException(status_code=500, detail="Error performing ML prediction.")


@app.post("/api/predict_dl")
async def predict_dl(payload: DLPredictionRequest):
    """
    Predicts disease type using a deep learning NLP model based on clinical text input.
    """
    if not dl_service:
        raise HTTPException(status_code=503, detail="DL Service not available.")

    try:
        result = await dl_service.predict(payload.note)
        return result
    except Exception as exc:
        logger.exception(f"DL Prediction Error: {exc}")
        raise HTTPException(status_code=500, detail="Error performing DL prediction.")


@app.post("/api/classify_image")
async def classify_image(file: UploadFile = File(...)):
    """
    Classifies skin conditions from an uploaded image using a CNN model.
    """
    if not image_service:
        raise HTTPException(status_code=503, detail="Image Classification Service not available.")

    try:
        image_bytes = await file.read()
        result = image_service.classify(image_bytes)
        return result
    except Exception as exc:
        logger.exception(f"Image Classification Error: {exc}")
        raise HTTPException(status_code=500, detail="Error classifying image.")


@app.post("/api/rag_chat")
async def rag_chat(payload: RAGQueryRequest):
    """
    Provides medical question answering using a Retrieval-Augmented Generation (RAG) model.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Service not available.")

    try:
        result_dict = rag_service.chat(payload.query)
        return {
            "response": result_dict.get("answer", "No answer found."),
            "sources": result_dict.get("sources", "Retrieved from local FAISS index."),
        }
    except Exception as exc:
        logger.exception(f"RAG Chat Error: {exc}")
        raise HTTPException(status_code=500, detail="Error processing RAG query.")
