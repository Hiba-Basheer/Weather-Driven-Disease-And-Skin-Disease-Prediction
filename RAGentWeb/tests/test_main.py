"""
test_main.py

Unit tests for the FastAPI application.

Covers:
- Application startup and service loading
- api endpoint behaviors
- Health check route
- Error handling
- Root HTML serving
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app, startup_event

# Global FastAPI test client
client = TestClient(app)


# Fixture: Mock heavy service imports (ML, DL, Image, RAG)
@pytest.fixture(autouse=True)
def mock_service_imports(monkeypatch):
    """
    Replace actual service classes in main.py with lightweight mocks.
    Prevents expensive model loading during tests.
    """
    mock_ml = MagicMock()
    mock_ml.predict.return_value = {"prediction": "ML Mock", "status": "Success"}

    mock_dl = MagicMock()
    mock_dl.predict.return_value = {"prediction": "DL Mock", "status": "Success"}

    mock_image = MagicMock()
    mock_image.classify.return_value = {
        "prediction": "Eczema",
        "confidence": 0.95,
        "status": "Success",
    }

    mock_rag = MagicMock()
    mock_rag.chat.return_value = {"answer": "RAG Mock Answer", "sources": "mock"}

    # Patch service classes in main.py
    monkeypatch.setattr("src.main.MLService", lambda *a, **k: mock_ml)
    monkeypatch.setattr("src.main.DLService", lambda *a, **k: mock_dl)
    monkeypatch.setattr(
        "src.main.ImageClassificationService", lambda *a, **k: mock_image
    )
    monkeypatch.setattr("src.main.RAGService", lambda *a, **k: mock_rag)

    # Inject mocks into global variables
    from src import main

    main.ml_service = mock_ml
    main.dl_service = mock_dl
    main.image_service = mock_image
    main.rag_service = mock_rag


# Fixture: Ensure startup event runs before tests
@pytest.fixture(autouse=True)
def run_startup():
    """Force the FastAPI startup event to initialize mocked services."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(startup_event())
    loop.close()


# Health Check Endpoint
def test_health_check():
    """Verify that /health returns a healthy status and all services are loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert all(data["services"].values())


# ML Prediction Endpoint
def test_predict_ml_endpoint():
    """Test /api/predict_ml with a valid payload."""
    payload = {
        "age": 35,
        "gender": "male",
        "city": "Mumbai",
        "symptoms": "fever, headache",
    }
    response = client.post("/api/predict_ml", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "ML Mock"
    assert result["status"] == "Success"


# DL Prediction Endpoint
def test_predict_dl_endpoint():
    """Test /api/predict_dl with a mock text input."""
    payload = {"note": "Age 40 male symptoms: severe headache, vomiting"}
    response = client.post("/api/predict_dl", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "DL Mock"
    assert result["status"] == "Success"


# Image Classification Endpoint
def test_classify_image_endpoint():
    """Test /api/classify_image with an uploaded dummy image."""
    from io import BytesIO

    from PIL import Image

    image = Image.new("RGB", (50, 50), color="red")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/classify_image",
        files={"file": ("test.jpg", buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "Eczema"
    assert result["confidence"] == 0.95
    assert result["status"] == "Success"


# RAG Chat Endpoint
def test_rag_chat_endpoint():
    """Test /api/rag_chat endpoint for a valid query."""
    payload = {"query": "What is dengue?"}
    response = client.post("/api/rag_chat", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["response"] == "RAG Mock Answer"
    assert "sources" in result


# Error Handling
def test_ml_service_unavailable(monkeypatch):
    """Ensure 503 is returned if ML service is missing."""
    from src import main

    main.ml_service = None  # Simulate missing service

    payload = {"age": 30, "gender": "female", "city": "Delhi", "symptoms": "cough"}
    response = client.post("/api/predict_ml", json=payload)

    assert response.status_code == 503
    assert "ML Service not available" in response.json()["detail"]


# Root HTML Serving
def test_root_returns_html():
    """Verify that the root (/) returns a valid HTML response."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"<!DOCTYPE html>" in response.content or b"<html" in response.content
