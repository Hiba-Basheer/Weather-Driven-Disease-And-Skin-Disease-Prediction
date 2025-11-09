# RAGentWeb/tests/test_dl_service.py
import os
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.dl_service import DLService

@pytest.fixture
def fake_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    dl_dir = models_dir / "dl"
    dl_dir.mkdir()

    import zipfile
    dl_model_path = dl_dir / "dl_model.keras"
    with zipfile.ZipFile(dl_model_path, 'w') as zf:
        zf.writestr("config.json", '{"class_name": "Model"}')
        zf.writestr("model.weights.bin", b'\x00' * 100)

    for f in ["scaler.pkl", "label_encoder.pkl"]:
        (dl_dir / f).touch()

    (dl_dir / "text_vectorizer_config.json").write_text("{}")
    (dl_dir / "text_vectorizer_vocab.txt").write_text("")

    os.environ["MODEL_ROOT"] = str(models_dir)
    return models_dir

@patch("src.dl_service.requests.get")
@patch("src.dl_service.tf.keras.models.load_model")
@patch("src.dl_service.joblib.load")
def test_dl_service_init(mock_joblib, mock_load_model, mock_get, fake_models):
    mock_load_model.return_value = MagicMock()
    mock_joblib.return_value = MagicMock()

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"main": {"temp": 25.0}}
    mock_get.return_value = mock_resp

    service = DLService("fake_key")
    assert service.model is not None

@patch("src.dl_service.requests.get")
@patch("src.dl_service.tf.keras.models.load_model")
@patch("src.dl_service.joblib.load")
@pytest.mark.asyncio
async def test_dl_predict(mock_joblib, mock_load_model, mock_get, fake_models):
    mock_load_model.return_value = MagicMock()
    mock_joblib.return_value = MagicMock()

    service = DLService("fake_key")
    service.model.predict = MagicMock(return_value=np.array([[0.1, 0.9]]))
    service.label_encoder.classes_ = np.array(["Flu", "Cold"])

    result = await service.predict("fever")
    assert result["prediction"] == "Cold"