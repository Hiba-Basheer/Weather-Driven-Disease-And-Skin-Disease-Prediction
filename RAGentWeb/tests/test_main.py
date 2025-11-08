# RAGentWeb/tests/test_main.py
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock
from src.main import app, startup_event

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

# Mock services before startup
@pytest.fixture(autouse=True)
def mock_services(monkeypatch):
    # ML/DL mocks
    mock_ml = AsyncMock()
    mock_ml.predict.return_value = {"prediction": "ML Mock", "status": "Success"}

    mock_dl = AsyncMock()
    mock_dl.predict.return_value = {"prediction": "DL Mock", "status": "Success"}

    # Image Service mock
    mock_image = MagicMock()
    mock_image.classify.return_value = {
        "prediction": "Eczema",
        "confidence": 0.95,
        "status": "Success"
    }

    # RAG mock
    mock_rag = AsyncMock()
    mock_rag.chat.return_value = {"response": "RAG Mock Answer", "sources": ["doc1", "doc2"]}

    monkeypatch.setattr("src.main.MLService", lambda *a, **k: mock_ml)
    monkeypatch.setattr("src.main.DLService", lambda *a, **k: mock_dl)
    monkeypatch.setattr("src.main.ImageClassificationService", lambda *a, **k: mock_image)
    monkeypatch.setattr("src.main.RAGService", lambda *a, **k: mock_rag)

    from src import main
    main.ml_service = mock_ml
    main.dl_service = mock_dl
    main.image_service = mock_image
    main.rag_service = mock_rag

@pytest.fixture(autouse=True)
def run_startup():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(startup_event())
    loop.close()

@pytest.mark.asyncio
async def test_health_check(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert all(data["services"].values())

@pytest.mark.asyncio
async def test_predict_ml_endpoint(async_client):
    payload = {"age": 35, "gender": "male", "city": "Mumbai", "symptoms": "fever"}
    response = await async_client.post("/api/predict_ml", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "ML Mock"

@pytest.mark.asyncio
async def test_predict_dl_endpoint(async_client):
    payload = {"note": "Age 40 male symptoms: headache"}        
    response = await async_client.post("/api/predict_dl", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "DL Mock"

@pytest.mark.asyncio
async def test_classify_image_endpoint(async_client):
    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", (50, 50), color="red")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = await async_client.post(
        "/api/classify_image", files={"file": ("test.jpg", buffer, "image/jpeg")}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "Eczema"

@pytest.mark.asyncio
async def test_rag_chat_endpoint(async_client):
    payload = {"query": "What is dengue?"}
    response = await async_client.post("/api/rag_chat", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["response"] == "RAG Mock Answer"

@pytest.mark.asyncio
async def test_root_returns_html(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
