import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
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


def check_class_distribution(y):
    """
    Check and log the class distribution of the target variable.

    Args:
        y (pd.Series or np.ndarray): Target vector.

    Returns:
        dict: Dictionary with class counts.
    """
    class_counts = Counter(y)
    logger.info("Class distribution:")
    for class_label, count in sorted(class_counts.items()):
        percentage = (count / len(y)) * 100
        logger.info(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
    
    # Check for imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 2:
        logger.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        logger.warning("Consider using class weights or resampling techniques.")
    else:
        logger.info("Classes are relatively balanced.")
    
    return dict(class_counts)


def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets.

    Args:
        y (pd.Series or np.ndarray): Target vector.

    Returns:
        dict: Dictionary mapping class labels to their weights.
    """
    logger.info("Computing class weights for imbalance handling...")
    unique_classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y
    )
    weight_dict = dict(zip(unique_classes, class_weights))
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict


def balance_dataset(X, y, method='smote', random_state=42):
    """
    Balance the dataset using oversampling or undersampling techniques.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        method (str): Balancing method. Options: 'smote', 'oversample', 'undersample'.
                     Default: 'smote'
        random_state (int): Random state for reproducibility. Default: 42

    Returns:
        X_balanced (np.ndarray): Balanced feature matrix.
        y_balanced (np.ndarray): Balanced target vector.
    """
    logger.info(f"Balancing dataset using method: {method}")
    
    # Check class distribution before balancing
    logger.info("Class distribution before balancing:")
    check_class_distribution(y)
    
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Apply balancing method
    if method.lower() == 'smote':
        try:
            smote = SMOTE(random_state=random_state)
            X_balanced, y_balanced = smote.fit_resample(X_array, y_array)
            logger.info("Applied SMOTE oversampling")
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Falling back to random oversampling.")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_balanced, y_balanced = ros.fit_resample(X_array, y_array)
            logger.info("Applied Random Over-sampling")
    
    elif method.lower() == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = ros.fit_resample(X_array, y_array)
        logger.info("Applied Random Over-sampling")
    
    elif method.lower() == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = rus.fit_resample(X_array, y_array)
        logger.info("Applied Random Under-sampling")
    
    else:
        raise ValueError(f"Unknown balancing method: {method}. "
                        f"Choose from: 'smote', 'oversample', 'undersample'")
    
    # Check class distribution after balancing
    logger.info("Class distribution after balancing:")
    check_class_distribution(y_balanced)
    
    logger.info(f"Dataset balanced. Shape changed from {X_array.shape} to {X_balanced.shape}")
    
    return X_balanced, y_balanced