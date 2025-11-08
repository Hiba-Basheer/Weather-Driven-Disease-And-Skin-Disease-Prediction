# RAGentWeb/tests/conftest.py
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    dl_dir = models_dir / "dl"
    dl_dir.mkdir()
    ml_dir = models_dir / "ml"
    ml_dir.mkdir()

    # DL: Required files
    (dl_dir / "dl_model.keras").touch()
    (dl_dir / "scaler.pkl").touch()
    (dl_dir / "label_encoder.pkl").touch()
    (dl_dir / "text_vectorizer_config.json").write_text("{}")
    (dl_dir / "text_vectorizer_vocab.txt").write_text("")

    # ML: Required files
    (ml_dir / "trained_model.pkl").touch()
    (ml_dir / "label_encoder.pkl").touch()
    (ml_dir / "ml_expected_columns.pkl").touch()

    # Image
    (models_dir / "resnet_model.h5").touch()
    (models_dir / "class_labels.txt").write_text("label1\nlabel2")

    # Set env var for services
    os.environ["MODEL_ROOT"] = str(models_dir)
    return models_dir


@pytest.fixture(autouse=True)
def patch_dl_validation():
    with patch("src.dl_service.DLService._validate_model_files"):
        yield
