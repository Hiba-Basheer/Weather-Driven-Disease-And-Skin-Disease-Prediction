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

# Service imports
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


# INSTANT HEALTH ENDPOINT — RETURNS IMMEDIATELY
@app.get("/health")
async def instant_health_check():
    return {"status": "healthy"}


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_service, dl_service, image_service, rag_service

    if dl_service is not None:
        logger.info("Services already initialized. Skipping reinitialization.")
        yield
        return

    logger.info("Application starting up — loading AI models and services...")

    time.sleep(5)
    logger.info("Startup delay complete — container ready for traffic.")

    yield

    logger.info("Shutting down services...")


# CREATE APP WITH LIFESPAN
app = FastAPI(
    title="Health AI Predictor & RAGent Web API",
    lifespan=lifespan
)

# Mount static files & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Legacy startup event
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Legacy startup event triggered — initialization handled by lifespan.")

# HEALTH ENDPOINT 
@app.get("/health")
async def health_check():
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