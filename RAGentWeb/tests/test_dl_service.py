from unittest.mock import MagicMock, patch
import pytest
from src.dl_service import DLService


@patch("src.dl_service.os.getenv")
@patch("src.dl_service.requests.get")
def test_dl_service_init(mock_get, mock_getenv, mock_model_files):
    """Test DLService initialization with mocked dependencies."""
    mock_getenv.return_value = "fake_key"
    with patch("src.dl_service.DLService.MODEL_DIR", mock_model_files):
        service = DLService()
        assert service.model is not None
        assert service.scaler is not None
        assert service.label_encoder is not None
        assert service.text_vectorizer is not None


@patch("src.dl_service.requests.get")
@pytest.mark.asyncio
async def test_dl_predict(mock_get, mock_model_files):
    """Test the predict method of DLService using mocked weather data."""
    # Mock weather API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 25.0, "humidity": 70},
        "wind": {"speed": 5.0},
    }
    mock_get.return_value = mock_response

    # Initialize DLService with mock model files
    with patch("src.dl_service.DLService.MODEL_DIR", mock_model_files):
        service = DLService(openweather_api_key="fake_key")
        result = await service.predict(
            note="Age 40 male symptoms: severe headache, vomiting",
            city="Mumbai",
        )
        assert "prediction" in result
        assert "confidence" in result
        assert "weather" in result
