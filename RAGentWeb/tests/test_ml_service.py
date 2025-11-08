import joblib
import numpy as np
import pytest
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.ml_service import MLService


@pytest.fixture
def mock_model_files(tmp_path):
    """
    Create temporary mock model files for MLService tests.
    Includes a real minimal trained model, label encoder, and expected feature list.
    """
    model_dir = tmp_path / "ml"
    model_dir.mkdir()

    # Minimal real model
    model_path = model_dir / "trained_model.pkl"
    # Minimal RandomForestClassifier
    mock_model = RandomForestClassifier(n_estimators=1, random_state=42)
    # Fit on minimal data to allow predict_proba
    mock_model.fit([[0, 0, 0, 0, 0, 0, 0]], [0])
    joblib.dump(mock_model, model_path)

    # Minimal real label encoder
    le_path = model_dir / "label_encoder.pkl"
    mock_label_encoder = LabelEncoder()
    mock_label_encoder.fit([0])  # label for the minimal model
    joblib.dump(mock_label_encoder, le_path)

    # Expected feature list
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
@pytest.mark.asyncio
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
    assert "prediction" in result
    assert "confidence" in result
    assert "raw_weather_data" in result
