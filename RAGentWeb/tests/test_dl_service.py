from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pytest
from src.dl_service import DLService

def fake_scaler(X):
    return X

@pytest.fixture(autouse=True)
def create_fake_scaler(fake_models: Path):
    scaler_path = fake_models / "dl" / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fake_scaler, scaler_path)


@patch("src.dl_service.os.getenv")
@patch("src.dl_service.requests.get")
def test_dl_service_init(mock_get, mock_getenv, fake_models):
    mock_getenv.return_value = "fake_key"
    with patch("src.dl_service.DLService.MODEL_DIR", str(fake_models / "dl")):
        service = DLService()
        assert service.model is not None


@patch("src.dl_service.requests.get")
@pytest.mark.asyncio
async def test_dl_predict(mock_get, fake_models):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 25.0, "humidity": 70},
        "wind": {"speed": 5.0},
    }
    mock_get.return_value = mock_response

    with patch("src.dl_service.DLService.MODEL_DIR", str(fake_models / "dl")):
        service = DLService(openweather_api_key="fake_key")
        result = await service.predict(
            note="Age 40 male symptoms: severe headache, vomiting",
            city="Mumbai",
        )
        assert "prediction" in result
