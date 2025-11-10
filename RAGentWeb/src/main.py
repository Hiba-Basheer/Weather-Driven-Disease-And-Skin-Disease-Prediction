"""
main.py
FastAPI backend for the Health AI Predictor & RAGent Web API.

Provides unified endpoints for:
  • ML-based disease prediction (structured input + weather)
  • DL-based disease prediction (text input)
  • Image-based skin disease classification
  • RAG-based medical question answering

All services are loaded on startup and exposed through REST endpoints.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGentWeb_API")

# Environment setup
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY") or ""
if not OPENWEATHER_API_KEY:
    logger.error("OPENWEATHERMAP_API_KEY not found in environment variables!")

# FastAPI initialization
BASE_DIR = Path(__file__).resolve().parent.parent

# Global service instances
ml_service: MLService | None = None
dl_service: DLService | None = None
image_service: ImageClassificationService | None = None
rag_service: RAGService | None = None


# Pydantic request models
class MLPredictionRequest(BaseModel):
    """Request schema for ML-based prediction."""
    age: int
    gender: str
    city: str
    symptoms: str


class DLPredictionRequest(BaseModel):
    """Request schema for DL-based text prediction."""
    note: str


class RAGQueryRequest(BaseModel):
    """Request schema for RAG-based question answering."""
    query: str


# Lifespan: actual model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_service, dl_service, image_service, rag_service

    if dl_service is not None:
        logger.info("Services already initialized. Skipping reinitialization.")
        yield
        return

    logger.info("Application starting up — loading AI models and services...")

    # ML Service
    try:
        ml_model_dir = BASE_DIR / "models" / "ml"
        if not ml_model_dir.exists():
            raise FileNotFoundError(f"ML model directory not found: {ml_model_dir}")
        ml_service = MLService(str(ml_model_dir), OPENWEATHER_API_KEY)
        logger.info("ML Service initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing ML Service: {e}")

    # DL Service
    try:
        dl_service = DLService(OPENWEATHER_API_KEY)
        logger.info("DL Service initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing DL Service: {e}")

    # Image Classification Service
    try:
        model_path = str(BASE_DIR / "models" / "resnet_model.h5")
        labels_path = str(BASE_DIR / "models" / "class_labels.txt")
        image_service = ImageClassificationService(model_path, labels_path)
        logger.info("Image Classification Service initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Image Service: {e}")

    # RAG Service
    try:
        faiss_path = str(BASE_DIR / "data" / "vector_store" / "faiss_index")
        rag_service = RAGService(faiss_path)
        logger.info(f"RAG Service initialized successfully from: {faiss_path}")
    except Exception as e:
        logger.error(f"Error initializing RAG Service: {e}")

    logger.info("All service initialization attempts complete.\n")

    time.sleep(5)
    logger.info("Startup delay complete — container ready for traffic.")

    yield

    logger.info("Shutting down services...")


# CREATE APP FIRST
app = FastAPI(
    title="Health AI Predictor & RAGent Web API",
    lifespan=lifespan
)

# Mount static files & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# INSTANT HEALTH ENDPOINT
@app.get("/health")
async def instant_health_check():
    return {"status": "healthy"}


# Legacy startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Legacy startup event — actual init in lifespan()."""
    logger.info("Legacy startup event triggered — initialization handled by lifespan.")


# Frontend route
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main frontend page (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})


# API Endpoints
@app.post("/api/predict_ml")
async def predict_ml_endpoint(payload: MLPredictionRequest):
    """Predicts disease using the ML model (structured + weather-based)."""
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
    except Exception as e:
        logger.error(f"ML Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict_dl")
async def predict_dl_endpoint(payload: DLPredictionRequest):
    """Predicts disease using the DL (NLP) model."""
    if not dl_service:
        raise HTTPException(status_code=503, detail="DL Service not available.")
    try:
        result = await dl_service.predict(payload.note)
        return result
    except Exception as e:
        logger.error(f"DL Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify_image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """Classifies an uploaded image using the CNN-based skin disease model."""
    if not image_service:
        raise HTTPException(status_code=503, detail="Image Classification Service not available.")
    try:
        image_bytes = await file.read()
        result = image_service.classify(image_bytes)
        return result
    except Exception as e:
        logger.error(f"Image Classification Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag_chat")
async def rag_chat_endpoint(payload: RAGQueryRequest):
    """Answers a health-related question using RAG (Retrieval-Augmented Generation)."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Chat Service not available.")
    try:
        result_dict = rag_service.chat(payload.query)
        response = result_dict.get("answer", "No answer found.")
        sources = result_dict.get("sources", "Data retrieved from local FAISS index.")
        return JSONResponse(content={"response": response, "sources": sources})
    except Exception as e:
        logger.error(f"RAG Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DETAILED HEALTH ENDPOINT
@app.get("/health")
async def health_check():
    """Detailed health check showing service status."""
    status = {
        "status": "healthy",
        "services": {
            "ml": ml_service is not None,
            "dl": dl_service is not None,
            "image": image_service is not None,
            "rag": rag_service is not None,
        },
        "message": "RAGentWeb API is running and all services are loaded.",
    }
    return JSONResponse(content=status)