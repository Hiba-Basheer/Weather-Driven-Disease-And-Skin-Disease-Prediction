import json
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest
import tensorflow as tf
from src.dl_service import DLService
from tensorflow.keras import layers


@pytest.fixture
def mock_model_files(tmp_path):
    """
    Create temporary mock model files for testing the DLService class.
    Includes a minimal Keras model, scaler, label encoder, and text vectorizer assets.
    """
    model_dir = tmp_path / "dl"
    model_dir.mkdir()

    # Mock Keras model
    model_path = model_dir / "dl_model.keras"
    num_input = layers.Input(shape=(71,), name="numeric_input")
    txt_input = layers.Input(shape=(100,), name="text_input")
    x_txt = layers.Embedding(200, 16)(txt_input)
    x_txt = layers.GlobalAveragePooling1D()(x_txt)
    merged = layers.concatenate([num_input, x_txt])
    output = layers.Dense(10, activation="softmax")(merged)
    model = tf.keras.models.Model([num_input, txt_input], output)
    model.save(model_path)

    # Mock scaler
    scaler = Mock()
    scaler.transform.return_value = np.zeros((1, 5))
    joblib.dump(scaler, model_dir / "scaler.pkl")

    # Mock label encoder
    label_encoder = Mock()
    label_encoder.classes_ = np.array([f"disease_{i}" for i in range(10)])
    label_encoder.inverse_transform.side_effect = lambda idx: np.array(
        [f"disease_{idx[0]}"]
    )
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")

    # Mock text vectorizer config
    with open(model_dir / "text_vectorizer_config.json", "w") as f:
        json.dump({"output_sequence_length": 100}, f)

    # Mock vocabulary file
    with open(model_dir / "text_vectorizer_vocab.txt", "w") as f:
        f.write("\n".join(["[UNK]", "fever", "headache"]))

    return str(model_dir)


@patch("src.dl_service.os.getenv")
@patch("src.dl_service.requests.get")
def test_dl_service_init(mock_get, mock_getenv, mock_model_files):
    """
    Test DLService initialization with mocked dependencies.
    Ensures that the model, scaler, label encoder, and text vectorizer are properly loaded.
    """
    mock_getenv.return_value = "fake_key"
    service = DLService()
    assert service.model is not None
    assert service.scaler is not None
    assert service.label_encoder is not None
    assert service.text_vectorizer is not None


@patch("src.dl_service.requests.get")
async def test_dl_predict(mock_get, mock_model_files):
    """
    Test the predict method of DLService using mocked weather data and inputs.
    Validates the structure and content of the prediction response.
    """
    # Mock weather API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "main": {"temp": 25.0, "humidity": 70},
        "wind": {"speed": 5.0},
    }
    mock_get.return_value = mock_response

    # Initialize DLService with mock model files
    with patch("src.dl_service.DLService.MODEL_DIR", mock_model_files):
        service = DLService(openweather_api_key="fake_key")

    # Run prediction
    result = await service.predict("Age 30 male from TestCity symptoms: fever headache")

    # Validate response structure
    assert result["status"] == "Success"
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["top_5"], list)
    assert len(result["top_5"]) == 5
