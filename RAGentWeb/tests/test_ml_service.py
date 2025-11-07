from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest

from src.ml_service import MLService


@pytest.fixture
def mock_model_files(tmp_path):
    """
    Create temporary mock model files for MLService tests.
    Includes a mock model, label encoder, and expected feature list.
    """
    model_dir = tmp_path / "ml"
    model_dir.mkdir()

    # Mock trained model
    model_path = model_dir / "trained_model.pkl"
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
    joblib.dump(mock_model, model_path)

    # Mock label encoder
    le_path = model_dir / "label_encoder.pkl"
    mock_label_encoder = Mock()
    mock_label_encoder.inverse_transform.return_value = np.array(["disease_1"])
    joblib.dump(mock_label_encoder, le_path)

    # Mock expected feature list
    features_path = model_dir / "ml_expected_columns.pkl"
    features = [
        "Age",
        "Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)",
        "fever",
        "headache",
        "gender",
    ]
    joblib.dump(features, features_path)

    return str(model_dir)


@patch("src.ml_service.fetch_and_log_weather_data")
async def test_ml_predict(mock_fetch_weather, mock_model_files):
    """
    Test the MLService.predict method with mocked weather data and input features.
    Ensures the prediction result structure and values are correct.
    """
    # Mock weather data response
    mock_fetch_weather.return_value = {"temp": 25.0, "humidity": 70, "wind_speed": 5.0}

    # Initialize MLService with mock model directory
    service = MLService(model_dir=mock_model_files, api_key="fake_key")

    # Test user input
    user_input = {
        "age": 30,
        "gender": "male",
        "city": "TestCity",
        "symptoms": "fever,headache",
    }

    # Perform prediction
    result = await service.predict(user_input)

    # Validate output
    assert result["status"] == "Success"
    assert result["prediction"] == "disease_1"
    assert result["confidence"] == 0.8
    assert "raw_weather_data" in result
