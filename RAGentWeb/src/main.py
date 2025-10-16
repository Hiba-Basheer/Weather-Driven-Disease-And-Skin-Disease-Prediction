import os
import logging
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path 


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGentWeb_API")

load_dotenv()  
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
if not OPENWEATHER_API_KEY:
    logger.error("OpenWeatherMap API key not found in environment variables!")
    
# Use pathlib for BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize FastAPI
app = FastAPI(title="Health AI Predictor & RAGent Web API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Service Imports
from .ml_service import MLService
from .dl_service import DLService
from .image_service import ImageClassificationService
from .rag_service import RAGService 

# Global service instances
ml_service: MLService = None
dl_service: DLService = None
image_service: ImageClassificationService = None
rag_service: RAGService = None


# Pydantic Models

class MLPredictionRequest(BaseModel):
    """Structured input for ML prediction."""
    age: int
    gender: str
    city: str
    symptoms: str

class DLPredictionRequest(BaseModel):
    """Free-text input for DL prediction."""
    note: str 

class RAGQueryRequest(BaseModel):
    """JSON input for RAG Chat query."""
    query: str


# Startup Event

@app.on_event("startup")
async def startup_event():
    """Initializes models and services upon application startup."""
    global ml_service, dl_service, image_service, rag_service
    logger.info("Application starting up. Loading models and services...")

    # ML Service
    try:
        ml_model_dir = BASE_DIR / "models" / "ml" 
        if not ml_model_dir.exists():
            raise FileNotFoundError(f"ML model not found at {ml_model_dir}")
        
        ml_service = MLService(str(ml_model_dir), OPENWEATHER_API_KEY)
        logger.info("ML Model (.pkl) loaded successfully.")
    except Exception as e:
        logger.error(f"Error initializing ML Service: {e}")
    # DL Service
    try:
        model_path = str(BASE_DIR / "models" / "dl" / "dl_model.keras")
        dl_service = DLService(OPENWEATHER_API_KEY)
        logger.info("Keras Model (dl_model.keras) loaded successfully.")
    except Exception as e:
        logger.error(f"Error initializing DL Service: {e}")

    # Image Classification Service
    try:
        model_path = str(BASE_DIR / "models" / "resnet_model.h5")
        labels_path = str(BASE_DIR / "models" / "class_labels.txt")
        image_service = ImageClassificationService(model_path, labels_path)
        logger.info("Image Classification Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error initializing Image Service: {e}")

    # RAG Chat Service
    try:
        faiss_path = str(BASE_DIR / "data" / "vector_store" / "faiss_index") 
        rag_service = RAGService(faiss_path)
        logger.info(f"RAG Service initialized successfully from FAISS path: {faiss_path}")
    except Exception as e:
        logger.error(f"Error initializing RAG Service (FAISS): {e}")

    logger.info("All startup initialization attempts complete.")


# Frontend

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


# ML Prediction Endpoint

@app.post("/api/predict_ml")
async def predict_ml_endpoint(payload: MLPredictionRequest):
    """ML prediction using structured input fields."""
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML Service not available.")
    try:
        result = await ml_service.predict({
            "age": payload.age,
            "gender": payload.gender,
            "city": payload.city,
            "symptoms": payload.symptoms
        })
        return result
    except Exception as e:
        logger.error(f"ML Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DL Prediction Endpoint

@app.post("/api/predict_dl")
async def predict_dl_endpoint(payload: DLPredictionRequest):
    """DL prediction using free-text input."""
    if not dl_service:
        raise HTTPException(status_code=503, detail="DL Service not available.")
    try:
        result = await dl_service.predict(payload.note) 
        return result
    except Exception as e:
        logger.error(f"DL Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Image Classification Endpoint

@app.post("/api/classify_image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """Classifies uploaded image."""
    if not image_service:
        raise HTTPException(status_code=503, detail="Image Classification Service not available.")
    try:
        image_bytes = await file.read()
        result = image_service.classify(image_bytes)
        return result
    except Exception as e:
        logger.error(f"Image Classification Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG Chat Endpoint

@app.post("/api/rag_chat")
async def rag_chat_endpoint(payload: RAGQueryRequest):
    """Handles free-text queries related to diseases or health topics."""
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