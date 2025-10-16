import os
import logging
import json
import shutil
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import joblib

import mlflow
import mlflow.tensorflow
from mlflow.tensorflow import MlflowCallback
from mlflow.models.signature import infer_signature

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths & Constants
BASE_DIR = r"D:\brototype\week27\DL"
DATA_PATH = r"D:/brototype/week27/DL/DL_module/dl_codes/data/generated_data.csv"
MLFLOW_EXPERIMENT = "DL_Module_Experiment"

_MLFLOW_BASE_URI = "file:///" + BASE_DIR.replace("\\", "/")
MLFLOW_TRACKING_URI = f"{_MLFLOW_BASE_URI}/mlruns"
MLFLOW_ARTIFACT_LOCATION = f"{_MLFLOW_BASE_URI}/mlflow_artifacts"
LOCAL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODELS_DIR = r"D:\brototype\week27\DL\DL_module\dl_codes\models"  


def ensure_directories() -> None:
    """
    Create required local directories for MLflow and custom artifacts.
    """
    for p in [os.path.join(BASE_DIR, "mlruns"),
              os.path.join(BASE_DIR, "mlflow_artifacts"),
              LOCAL_ARTIFACTS_DIR,
              MODELS_DIR]:
        os.makedirs(p, exist_ok=True)
    logger.info("Ensured directories exist under %s", BASE_DIR)


def init_mlflow(experiment_name: str) -> None:
    """
    Initialize MLflow with a local tracking URI and ensure the experiment exists with
    the desired artifact location. 

    Args:
        experiment_name: Name of the MLflow experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Try to get the experiment by name; create with custom artifact path if missing.
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        # Create experiment with artifact_location 
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
        logger.info("Created MLflow experiment '%s' (id=%s) with artifacts at %s",
                    experiment_name, exp_id, MLFLOW_ARTIFACT_LOCATION)
    else:
        logger.info("Using existing MLflow experiment '%s' (id=%s). Artifacts: %s",
                    existing.name, existing.experiment_id, existing.artifact_location)

    mlflow.set_experiment(experiment_name)
    logger.info("Tracking URI: %s", MLFLOW_TRACKING_URI)


def log_gpu_info() -> None:
    """
    Log TensorFlow GPU availability and basic device info, plus `nvidia-smi` output (if available).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPU detected. TensorFlow will run on CPU.")
        mlflow.log_param("tf_gpu_available", False)
        return

    mlflow.log_param("tf_gpu_available", True)
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    details = tf.config.experimental.get_device_details(gpus[0])
    gpu_name = details.get("device_name", "GPU:0")
    logger.info("Detected GPU: %s", gpu_name)
    mlflow.log_param("gpu_0_name", gpu_name)

    # LLog nvidia-smi output as an artifact for deeper debugging
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output(["nvidia-smi", "-q"], text=True, stderr=subprocess.STDOUT)
            gpu_info_txt = os.path.join(LOCAL_ARTIFACTS_DIR, "nvidia_smi.txt")
            with open(gpu_info_txt, "w", encoding="utf-8") as f:
                f.write(smi)
            mlflow.log_artifact(gpu_info_txt)
        except Exception as e:
            logger.debug("Could not run nvidia-smi: %s", e)


def save_training_curves(history, out_png_path: str) -> None:
    """
    Save accuracy and loss curves to a PNG file.

    Args:
        history: Keras History object.
        out_png_path: Path to save the PNG file.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path)
    plt.close()
    logger.info("Performance metrics plot saved to: %s", out_png_path)


def create_and_train_model() -> None:
    """
    Full pipeline:
    - Train DL model and save all artifacts (model, preprocessors, etc.)
      into D:\brototype\week27\DL\DL_module\dl_codes\models
    """
    try:
        ensure_directories()
        init_mlflow(MLFLOW_EXPERIMENT)

        # Load data
        df = pd.read_csv(DATA_PATH)
        logger.info("Dataset loaded with shape: %s", df.shape)

        class_counts = Counter(df['prognosis'])
        single_instance_classes = {cls for cls, count in class_counts.items() if count <= 1}
        if single_instance_classes:
            df = df[~df['prognosis'].isin(single_instance_classes)]
            logger.warning("Removed classes with 1 or fewer instances: %s. New shape: %s",
                           single_instance_classes, df.shape)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['prognosis'])

        numeric_features = ['Age', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'symptom_count']
        X_numeric_raw = df[numeric_features].values
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric_raw)

        text_features = df['symptom_profile'].astype(str)
        max_tokens = 5000
        max_len = 50

        text_vectorizer = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len)
        text_vectorizer.adapt(text_features.values)
        text_sequences = text_vectorizer(text_features.values).numpy()

        # Split data
        X_num_train, X_num_temp, X_text_train, X_text_temp, y_train, y_temp = train_test_split(
            X_numeric, text_sequences, y, test_size=0.3, random_state=42, stratify=y
        )
        X_num_val, X_num_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
            X_num_temp, X_text_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Handle class imbalance
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))

        X_train_combined = np.hstack((X_num_train, X_text_train))
        over = SMOTE(sampling_strategy='auto', random_state=42)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        pipeline = Pipeline(steps=[('o', over), ('u', under)])
        X_resampled_combined, y_resampled = pipeline.fit_resample(X_train_combined, y_train)

        num_cols = X_num_train.shape[1]
        X_resampled_num = X_resampled_combined[:, :num_cols]
        X_resampled_text = X_resampled_combined[:, num_cols:]

        # Build model
        num_input = layers.Input(shape=(len(numeric_features),), name='numeric_input')
        x_num = layers.Dense(128, activation='relu')(num_input)
        x_num = layers.Dropout(0.4)(x_num)
        x_num = layers.Dense(64, activation='relu')(x_num)

        text_input = layers.Input(shape=(max_len,), name='text_input')
        x_txt = layers.Embedding(input_dim=max_tokens, output_dim=128)(text_input)
        x_txt = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x_txt)
        x_txt = layers.GlobalMaxPooling1D()(x_txt)
        x_txt = layers.Dropout(0.4)(x_txt)

        merged = layers.concatenate([x_num, x_txt])
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.4)(x)
        output = layers.Dense(len(np.unique(y)), activation='softmax')(x)

        model = models.Model(inputs=[num_input, text_input], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=lambda s: logger.info(s))

        # === Training ===
        params = {
            'epochs': 50,
            'batch_size': 32,
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            log_gpu_info()

            # === Save preprocessors directly into MODELS_DIR ===
            os.makedirs(MODELS_DIR, exist_ok=True)

            scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_path)

            vectorizer_cfg_path = os.path.join(MODELS_DIR, "text_vectorizer_config.json")
            with open(vectorizer_cfg_path, 'w', encoding='utf-8') as f:
                json.dump(text_vectorizer.get_config(), f)

            vocab_path = os.path.join(MODELS_DIR, "text_vectorizer_vocab.txt")
            with open(vocab_path, 'w', encoding='utf-8') as f:
                for word in text_vectorizer.get_vocabulary():
                    f.write(word + '\n')

            logger.info("Saved all preprocessors into models directory.")

            # === Training Callbacks ===
            ckpt_path = os.path.join(MODELS_DIR, "dl_model.keras")
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

            history = model.fit(
                [X_resampled_num, X_resampled_text],
                y_resampled,
                validation_data=([X_num_val, X_text_val], y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop, ckpt, reduce_lr],
                class_weight=class_weights_dict,
                verbose=1
            )

            # Save curves
            plot_path = os.path.join(MODELS_DIR, "performance_metrics.png")
            save_training_curves(history, plot_path)

            # Save final model
            model.save(ckpt_path)
            logger.info("Model and all artifacts saved to %s", MODELS_DIR)

    except Exception as e:
        logger.error("An error occurred: %s", e)


if __name__ == "__main__":
    create_and_train_model()