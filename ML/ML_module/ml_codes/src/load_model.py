import joblib

def load_trained_model(model_path, le_path):
    """Load trained model and label encoder."""
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le
