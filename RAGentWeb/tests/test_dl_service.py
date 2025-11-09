# RAGentWeb/tests/test_dl_service.py
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from src.dl_service import DLService

@pytest.fixture
def fake_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    dl_dir = models_dir / "dl"
    dl_dir.mkdir()

    # Create valid .keras ZIP
    import zipfile
    dl_model_path = dl_dir / "dl_model.keras"
    with zipfile.ZipFile(dl_model_path, 'w') as zf:
        zf.writestr("config.json", '{"class_name": "Model"}')
        zf.writestr("model.weights.bin", b'\x00' * 100)

    # Create fake pickle files (empty is fine)
    for f in ["scaler.pkl", "label_encoder.pkl"]:
        (dl_dir / f).touch()

    # Fake text vectorizer files
    (dl_dir / "text_vectorizer_config.json").write_text("{}")
    (dl_dir / "text_vectorizer_vocab.txt").write_text("")

    os.environ["MODEL_ROOT"] = str(models_dir)
    return models_dir

@patch("src.dl_service.tf.keras.models.load_model")
@patch("src.dl_service.joblib.load")
@patch("src.dl_service.requests.get")
def test_dl_service_init(mock_get, mock_joblib, mock_load_model, fake_models):
    mock_load_model.return_value = MagicMock()
    mock_joblib.side_effect = lambda path: MagicMock()  # Return fake scaler/encoder

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 25.0, "humidity": 70},
        "wind": {"speed": 5.0},
    }
    mock_get.return_value = mock_response

    service = DLService(openweather_api_key="fake_key")
    assert service.model is not None
    assert service.scaler is not None

@patch("src.dl_service.tf.keras.models.load_model")
@patch("src.dl_service.joblib.load")
@patch("src.dl_service.requests.get")
@pytest.mark.asyncio
async def test_dl_predict(mock_get, mock_joblib, mock_load_model, fake_models):
    mock_load_model.return_value = MagicMock()
    mock_joblib.side_effect = lambda path: MagicMock()

    service = DLService(openweather_api_key="fake_key")

    # Mock model predict
    service.model.predict = MagicMock(return_value=np.array([[0.1, 0.9]]))
    service.label_encoder.classes_ = np.array(["Flu", "Cold"])

    result = await service.predict("Age 30 male fever cough")
    assert result["prediction"] == "Cold"
    assert result["confidence"] > 0.8