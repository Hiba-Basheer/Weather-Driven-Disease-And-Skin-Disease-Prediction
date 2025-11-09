# RAGentWeb/tests/test_image_service.py
from unittest.mock import MagicMock, mock_open, patch
from PIL import Image
import io
import numpy as np
from src.image_service import ImageClassificationService

@patch("os.path.exists", return_value=True)
@patch("tensorflow.keras.models.load_model")
@patch("builtins.open", new_callable=mock_open, read_data="label1\nlabel2\nlabel3")
def test_image_service_init(mock_file, mock_load_model, mock_exists):
    mock_model = MagicMock()
    mock_model.input_shape = (None, 224, 224, 3)
    mock_load_model.return_value = mock_model

    service = ImageClassificationService("fake.h5", "fake_labels.txt")
    assert service.model is not None
    assert hasattr(service, "labels")
    assert len(service.labels) == 3
    assert service.target_size == (224, 224)


@patch("os.path.exists", return_value=True)
@patch("tensorflow.keras.models.load_model")
@patch("builtins.open", new_callable=mock_open, read_data="label1\nlabel2\nlabel3")
def test_image_classify(mock_file, mock_load_model, mock_exists):
    # Mock model with correct input shape
    mock_model = MagicMock()
    mock_model.input_shape = (None, 224, 224, 3)
    mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # label2 wins
    mock_load_model.return_value = mock_model

    service = ImageClassificationService("fake.h5", "fake_labels.txt")

    # Create proper RGB image
    img = Image.new("RGB", (224, 224), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    result = service.classify(buffer.read())

    assert result["prediction"] == "label2"
    assert result["confidence"] == 0.8
    assert result["status"] == "Success"
    assert len(result["top_predictions"]) == 3