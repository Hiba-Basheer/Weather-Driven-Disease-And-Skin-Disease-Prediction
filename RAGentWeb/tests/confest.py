# RAGentWeb/tests/conftest.py
import os
import pytest


# Fake model folder for DL & Image tests
@pytest.fixture
def fake_models(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "dl").mkdir()
    (models / "ml").mkdir()

    # DL files
    for f in [
        "dl_model.keras",
        "scaler.pkl",
        "label_encoder.pkl",
        "text_vectorizer_config.json",
        "text_vectorizer_vocab.txt",
    ]:
        (models / "dl" / f).touch()

    # ML files
    for f in [
        "trained_model.pkl",
        "label_encoder.pkl",
        "ml_expected_columns.pkl",
    ]:
        (models / "ml" / f).touch()

    # Image files
    (models / "resnet_model.h5").touch()
    (models / "class_labels.txt").write_text("cat\ndog\nbird")

    # Make the folder visible to the services
    os.environ["MODEL_ROOT"] = str(models)
    return models