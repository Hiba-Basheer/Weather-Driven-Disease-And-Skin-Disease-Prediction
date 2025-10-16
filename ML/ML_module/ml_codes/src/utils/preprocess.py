import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger.info(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    logger.info(f"Data loaded. Shape: {data.shape}")
    return data


def preprocess_data(data):
    """
    Clean and preprocess the data.

    - Drops missing values
    - Encodes categorical variables
    - Adds lag features

    Args:
        data (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Preprocessed data.
        LabelEncoder: Fitted label encoder.
    """
    logger.info("Starting preprocessing...")

    # Drop missing rows
    initial_shape = data.shape
    data = data.dropna()
    logger.info(f"Dropped missing values: {initial_shape[0] - data.shape[0]} rows removed.")

    # Encode 'Gender' and 'prognosis' columns
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"])
    data["prognosis"] = le.fit_transform(data["prognosis"])
    logger.info("Encoded categorical variables: Gender and prognosis.")

    # Create lag features
    data["Temperature_lag1"] = data["Temperature (C)"].shift(1).fillna(data["Temperature (C)"].mean())
    data["Humidity_lag1"] = data["Humidity"].shift(1).fillna(data["Humidity"].mean())
    data["WindSpeed_lag1"] = data["Wind Speed (km/h)"].shift(1).fillna(data["Wind Speed (km/h)"].mean())
    logger.info("Lag features created: Temperature_lag1, Humidity_lag1, WindSpeed_lag1.")

    logger.info("Preprocessing complete.")
    return data, le


def get_features(data):
    """
    Extract features and target from the dataset.

    Args:
        data (pd.DataFrame): Preprocessed dataset.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    """
    logger.info("Extracting features and target...")
    y = data["prognosis"]
    X = data.drop(columns=["prognosis"])
    X = X.loc[:, ~X.columns.duplicated()]
    logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
    return X, y