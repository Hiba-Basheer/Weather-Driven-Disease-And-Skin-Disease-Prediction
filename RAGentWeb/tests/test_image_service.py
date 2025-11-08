# RAGentWeb/tests/test_image_service.py
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
from src.image_service import ImageClassificationService


@patch("os.path.exists", return_value=True)
@patch("tensorflow.keras.models.load_model")
@patch("builtins.open", new_callable=mock_open, read_data="label1\nlabel2")
def test_image_service_init(mock_file, mock_load_model, mock_exists):
    service = ImageClassificationService("fake_model.h5", "fake_labels.txt")
    assert service.model is not None
    assert hasattr(service, "labels")
    assert service.labels == ["label1", "label2"]


@patch("os.path.exists", return_value=True)
@patch("tensorflow.keras.models.load_model")
@patch("builtins.open", new_callable=mock_open, read_data="label1\nlabel2")
def test_image_classify(mock_file, mock_load_model, mock_exists):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.9]])
    mock_load_model.return_value = mock_model

    service = ImageClassificationService("fake_model.h5", "fake_labels.txt")

    # Create real JPEG bytes
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (224, 224), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    result = service.classify(buffer.read())
    assert result["prediction"] == "label2"