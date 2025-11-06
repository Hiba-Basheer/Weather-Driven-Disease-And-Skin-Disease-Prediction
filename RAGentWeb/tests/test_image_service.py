import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, Mock
from src.image_service import ImageClassificationService


@pytest.fixture
def mock_model_files(tmp_path):
    """
    Create temporary mock model and label files for testing ImageClassificationService.
    Returns paths for a dummy model file and label file.
    """
    model_dir = tmp_path
    model_path = model_dir / "resnet_model.h5"
    labels_path = model_dir / "class_labels.txt"

    # Mock Keras model
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_model.input_shape = (None, 224, 224, 3)

    # Write dummy labels
    with open(labels_path, "w") as f:
        f.write("class1\nclass2\nclass3")

    # Patch model loading to return mock model
    with patch("tensorflow.keras.models.load_model", return_value=mock_model):
        yield str(model_path), str(labels_path)


def test_image_service_init(mock_model_files):
    """
    Test initialization of ImageClassificationService.
    Verifies model loading, target size, and label count.
    """
    model_path, labels_path = mock_model_files
    service = ImageClassificationService(model_path, labels_path)

    assert service.model is not None
    assert service.target_size == (224, 224)
    assert len(service.class_labels) == 3


def test_image_classify(mock_model_files):
    """
    Test the classify method of ImageClassificationService using a dummy image.
    Ensures correct prediction structure and values.
    """
    model_path, labels_path = mock_model_files
    service = ImageClassificationService(model_path, labels_path)

    # Create dummy image bytes
    image = Image.new("RGB", (224, 224), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Run classification
    result = service.classify(buffer.read())

    # Validate output structure and content
    assert result["status"] == "Success"
    assert result["prediction"] == "class1"
    assert result["confidence"] == 0.8
    assert isinstance(result["top_predictions"], list)
    assert len(result["top_predictions"]) == 3
