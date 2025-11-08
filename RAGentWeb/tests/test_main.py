import pytest
from httpx import AsyncClient
from src.main import app


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_predict_ml_endpoint(client):
    payload = {
        "age": 35,
        "gender": "male",
        "city": "Mumbai",
        "symptoms": "fever, headache",
    }
    response = await client.post("/api/predict_ml", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_predict_dl_endpoint(client):
    payload = {"note": "Age 40 male symptoms: severe headache, vomiting"}
    response = await client.post("/api/predict_dl", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_classify_image_endpoint(client):
    from io import BytesIO

    from PIL import Image

    image = Image.new("RGB", (50, 50), color="red")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = await client.post(
        "/api/classify_image",
        files={"file": ("test.jpg", buffer, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


@pytest.mark.asyncio
async def test_rag_chat_endpoint(client):
    payload = {"query": "What is dengue?"}
    response = await client.post("/api/rag_chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


@pytest.mark.asyncio
async def test_root_returns_html(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
