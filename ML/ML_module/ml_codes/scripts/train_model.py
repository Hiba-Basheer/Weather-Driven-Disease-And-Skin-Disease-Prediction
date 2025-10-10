"""
Model Training Module for Weather-Driven Disease Prediction

This script trains and evaluates two machine learning models—Random Forest and XGBoost—
on weather and symptom data to predict diseases. It selects the better-performing model
based on accuracy and recall, logs metrics using MLflow, and saves the trained model,
label encoder, and expected feature columns to disk for consistent inference.

Steps:
- Load and preprocess raw data
- Split data into train/test sets
- Train Random Forest and XGBoost models
- Evaluate performance
- Log metrics to MLflow
- Save the best model, encoder, and feature columns
"""

from pathlib import Path
import sys
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Append src path
current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from utils.preprocess import load_data, preprocess_data, get_features  

def train_model() -> None:
    """Train and evaluate ML models, log metrics, and save artifacts."""
    try:
        # Define paths
        base_dir = Path("D:/brototype/week27/ML/ML_module/ml_codes")
        data_path = base_dir / "data/raw/Weather-related disease prediction.csv"
        model_dir = base_dir / "models"
        processed_dir = base_dir / "data/processed"
        model_path = model_dir / "trained_model.pkl"
        label_path = model_dir / "label_encoder.pkl"
        feature_columns_path = model_dir / "ml_expected_columns.pkl"
        processed_data_path = processed_dir / "processed_dataset.csv"

        # Load and preprocess
        logger.info("Loading and preprocessing data...")
        data = load_data(str(data_path))
        data, label_encoder = preprocess_data(data)
        X, y = get_features(data)

        # Remove duplicate columns
        X = X.loc[:, ~X.columns.duplicated()]
        logger.info(f"Final feature count after deduplication: {X.shape[1]}")

        # Save processed dataset
        processed_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to: {processed_data_path}")

        # Save expected feature columns for inference
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(X.columns.tolist(), feature_columns_path)
        logger.info(f"Expected feature columns saved to: {feature_columns_path}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Start MLflow experiment
        mlflow.set_experiment("ML MODULE")
        with mlflow.start_run(run_name="Model_Comparison"):
            # Train Random Forest
            logger.info("Training Random Forest...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_recall = recall_score(y_test, rf_pred, average="macro")
            mlflow.log_metric("rf_accuracy", rf_acc)
            mlflow.log_metric("rf_recall", rf_recall)
            mlflow.sklearn.log_model(rf_model, artifact_path="random_forest")

            # Train XGBoost
            logger.info("Training XGBoost...")
            xgb_model = XGBClassifier(eval_metric="mlogloss", random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_recall = recall_score(y_test, xgb_pred, average="macro")
            mlflow.log_metric("xgb_accuracy", xgb_acc)
            mlflow.log_metric("xgb_recall", xgb_recall)
            mlflow.xgboost.log_model(xgb_model, artifact_path="xgboost")

            # Select best model
            best_model = xgb_model if xgb_acc > rf_acc else rf_model
            best_model_name = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
            logger.info(f"{best_model_name} selected as best model.")

            # Log model metadata
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("feature_count", X.shape[1])

            # Save model and encoder
            joblib.dump(best_model, model_path)
            joblib.dump(label_encoder, label_path)
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Label encoder saved to: {label_path}")

    except Exception as e:
        logger.exception("Training failed.")

if __name__ == "__main__":
    train_model()