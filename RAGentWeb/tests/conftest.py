# RAGentWeb/tests/conftest.py 
import pytest
from pathlib import Path
import zipfile
from unittest.mock import MagicMock, patch

@pytest.fixture
def fake_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    dl_dir = models_dir / "dl"
    dl_dir.mkdir()
    ml_dir = models_dir / "ml"
    ml_dir.mkdir()

    # DL: Create valid .keras ZIP
    dl_model_path = dl_dir / "dl_model.keras"
    with zipfile.ZipFile(dl_model_path, 'w') as zf:
        zf.writestr("config.json", '{"class_name": "Model"}')
        zf.writestr("model.weights.bin", b'\x00' * 100)

    (dl_dir / "scaler.pkl").touch()
    (dl_dir / "label_encoder.pkl").touch()
    (dl_dir / "text_vectorizer_config.json").write_text("{}")
    (dl_dir / "text_vectorizer_vocab.txt").write_text("")

    # ML
    (ml_dir / "trained_model.pkl").touch()
    (ml_dir / "label_encoder.pkl").touch()
    (ml_dir / "ml_expected_columns.pkl").touch()

    # Image
    (models_dir / "resnet_model.h5").touch()
    (models_dir / "class_labels.txt").write_text("label1\nlabel2")

    os.environ["MODEL_ROOT"] = str(models_dir)
    return models_dir

@pytest.fixture(autouse=True)
def mock_loaders():
    with patch("tensorflow.keras.models.load_model", return_value=MagicMock(input_shape=(1, 224, 224, 3))), \
         patch("src.rag_service.HuggingFaceEmbeddings", return_value=MagicMock()), \
         patch("src.rag_service.FAISS.load_local", return_value=MagicMock()):
        yield