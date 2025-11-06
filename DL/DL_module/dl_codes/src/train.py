"""
train.py
Train DL model with all structured features (5 continuous + 66 discrete) + text.
Implements the feature-balanced multimodal architecture.
Saves all preprocessors and the final model.
"""

import os
import logging
import json
import shutil
import subprocess
import re
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import joblib

import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

# LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# PATHS 
BASE_DIR = r"D:\brototype\week27\DL"
DATA_PATH = r"D:/brototype/week27/DL/DL_module/dl_codes/data/generated_data.csv"
MLFLOW_EXPERIMENT = "DL_Module_Experiment"

_MLFLOW_BASE_URI = "file:///" + BASE_DIR.replace("\\", "/")
MLFLOW_TRACKING_URI = f"{_MLFLOW_BASE_URI}/mlruns"
MLFLOW_ARTIFACT_LOCATION = f"{_MLFLOW_BASE_URI}/mlflow_artifacts"
LOCAL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = r"D:\brototype\week27\DL\DL_module\dl_codes\models"

# CONFIGURATION 
LSTM_UNITS = 128    
DENSE_UNITS = 16    
DROPOUT_RATE_NUM = 0.5 
DROPOUT_RATE_TEXT = 0.3 
MAX_SEQ_LENGTH = 100 
EMBEDDING_DIM = 128 

# Full list of symptom columns
SYMPTOM_COLUMNS = [
    'chills', 'loss_of_balance', 'reduced_smell_and_taste', 'nasal_polyps', 'pain_behind_eyes',
    'pain_radiating_to_left_arm', 'dry_skin', 'joint_pain', 'body_aches', 'upper_back_pain', 'joint_stiffness',
    'knee_ache', 'slurred_speech', 'back_pain', 'arm_pain', 'rapid_heart_rate', 'rapid_breathing',
    'throbbing_headache', 'diarrhea', 'high_cholesterol', 'vomiting', 'sinus_headache', 'sweating',
    'lightheadedness', 'shortness_of_breath', 'sore_throat', 'trouble_seeing', 'sensitivity_to_light',
    'trouble_speaking', 'weakness_in_arms_or_legs', 'swollen_glands', 'fever', 'anxiety', 'jaw_pain',
    'high_fever', 'weakness', 'nausea', 'dizziness', 'abdominal_pain', 'runny_nose', 'sneezing',
    'sensitivity_to_sound', 'sudden_numbness_on_one_side', 'facial_pain', 'bleeding_gums', 'fatigue',
    'loss_of_appetite', 'skin_irritation', 'headache', 'severe_headache', 'confusion', 'chest_pain',
    'diabetes', 'cough', 'hiv_aids', 'asthma_history', 'itchiness', 'asthma', 'jaw_discomfort',
    'high_blood_pressure', 'blurred_vision', 'rashes', 'obesity', 'shivering', 'pain_behind_the_eyes'
]
CONTINUOUS_COLS = ['Age', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'symptom_count']


def ensure_directories() -> None:
    for p in [os.path.join(BASE_DIR, "mlruns"), os.path.join(BASE_DIR, "mlflow_artifacts"), LOCAL_ARTIFACTS_DIR, MODELS_DIR]:
        os.makedirs(p, exist_ok=True)
    logger.info("Ensured directories exist under %s", BASE_DIR)


def init_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        exp_id = mlflow.create_experiment(name=experiment_name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, exp_id)
    else:
        logger.info("Using existing MLflow experiment '%s' (id=%s)", existing.name, existing.experiment_id)
    mlflow.set_experiment(experiment_name)


def log_gpu_info() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPU detected. Running on CPU.")
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
    try:
        ensure_directories()
        init_mlflow(MLFLOW_EXPERIMENT)

        # LOAD DATA
        df = pd.read_csv(DATA_PATH)
        logger.info("Dataset loaded with shape: %s", df.shape)

        # REMOVE LOW-COUNT CLASSES
        class_counts = Counter(df['prognosis'])
        single_instance_classes = {cls for cls, count in class_counts.items() if count <= 1}
        if single_instance_classes:
            df = df[~df['prognosis'].isin(single_instance_classes)]
            logger.warning("Removed classes with 1 or fewer instances: %s. New shape: %s", single_instance_classes, df.shape)

        # LABEL ENCODER
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['prognosis'])

        # DATA PREPARATION 
        
        # Identify all binary/one-hot symptom columns + Gender
        binary_cols = [col for col in SYMPTOM_COLUMNS if col in df.columns and col not in CONTINUOUS_COLS]
        binary_cols.append('Gender') 
        
        # 1. Scale ONLY the continuous features
        X_continuous_raw = df[CONTINUOUS_COLS].fillna(0).values
        scaler = StandardScaler()
        X_continuous_scaled = scaler.fit_transform(X_continuous_raw) 
        
        # 2. Combine Scaled Continuous Features with Raw Binary/Discrete Features
        X_binary_raw = df[binary_cols].fillna(0).values
        X_numeric = np.hstack([X_continuous_scaled, X_binary_raw])
        
        logger.info("Structured Input finalized with shape: %s (%s continuous + %s binary/discrete)", 
                    X_numeric.shape, len(CONTINUOUS_COLS), len(binary_cols))
        
        # TEXT FEATURES
        text_features = df['text'].astype(str)
        max_len = MAX_SEQ_LENGTH

        text_vectorizer = layers.TextVectorization(max_tokens=None, output_sequence_length=max_len)
        text_vectorizer.adapt(text_features.values)
        text_sequences = text_vectorizer(text_features.values).numpy()
        vocab_size = text_vectorizer.vocabulary_size()
        logger.info("Text Vectorizer adapted (Vocab Size: %s)", vocab_size)

        # SPLIT
        X_num_train, X_num_temp, X_text_train, X_text_temp, y_train, y_temp = train_test_split(
            X_numeric, text_sequences, y, test_size=0.3, random_state=42, stratify=y
        )
        X_num_val, X_num_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
            X_num_temp, X_text_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # CLASS WEIGHTS
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        logger.info("Using Class Weights for imbalance mitigation.")

        X_train_num_final = X_num_train
        X_train_text_final = X_text_train
        y_train_final = y_train


        # FEATURE BALANCING 
        num_structured_features = X_numeric.shape[1]

        # 1. Structured (Numeric/Binary) Input Branch
        num_input = layers.Input(shape=(num_structured_features,), name='numeric_input')
        # Dampen the influence of structured features (DENSE_UNITS=16, DROPOUT_RATE_NUM=0.5)
        x_num = layers.Dense(DENSE_UNITS, activation='relu')(num_input) 
        x_num = layers.Dropout(DROPOUT_RATE_NUM)(x_num)

        # 2. Text Input Branch
        text_input = layers.Input(shape=(max_len,), name='text_input')
        x_txt = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(text_input) 
        # Enhanced capacity (LSTM_UNITS=128)
        x_txt = layers.Bidirectional(layers.LSTM(LSTM_UNITS))(x_txt) 
        x_txt = layers.Dropout(DROPOUT_RATE_TEXT)(x_txt) 

        # 3. Concatenate and Output
        merged = layers.concatenate([x_num, x_txt])
        x = layers.Dense(64, activation='relu')(merged)
        x = layers.Dropout(0.4)(x)
        output = layers.Dense(len(np.unique(y)), activation='softmax')(x)

        model = models.Model(inputs=[num_input, text_input], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=lambda s: logger.info(s))

        # TRAINING
        params = {'epochs': 50, 'batch_size': 32, 'lstm_units': LSTM_UNITS, 'numeric_dense_units': DENSE_UNITS, 'numeric_dropout': DROPOUT_RATE_NUM}

        with mlflow.start_run():
            mlflow.log_params(params)
            log_gpu_info()

            # SAVE TO MODELS_DIR
            os.makedirs(MODELS_DIR, exist_ok=True)

            # Save scaler (5 continuous features)
            scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            # Save label encoder
            label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_path)

            # Save text vectorizer 
            vectorizer_cfg_path = os.path.join(MODELS_DIR, "text_vectorizer_config.json")
            with open(vectorizer_cfg_path, 'w', encoding='utf-8') as f:
                json.dump(text_vectorizer.get_config(), f)

            vocab_path = os.path.join(MODELS_DIR, "text_vectorizer_vocab.txt")
            with open(vocab_path, 'w', encoding='utf-8') as f:
                for word in text_vectorizer.get_vocabulary():
                    f.write(word + '\n')
            
            logger.info("Saved all preprocessors into models directory.")

            # CALLBACKS
            ckpt_path = os.path.join(MODELS_DIR, "dl_model.keras")
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

            history = model.fit(
                [X_train_num_final, X_train_text_final],
                y_train_final,
                validation_data=([X_num_val, X_text_val], y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop, ckpt, reduce_lr],
                class_weight=class_weights_dict,
                verbose=1
            )

            # EVALUATE
            y_pred = np.argmax(model.predict([X_num_test, X_text_test], verbose=0), axis=1)
            accuracy = np.mean(y_pred == y_test)
            cm = confusion_matrix(y_test, y_pred)

            # CONFUSION MATRIX
            fig, ax = plt.subplots(figsize=(10, 8))
            prognosis_names = label_encoder.classes_
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=prognosis_names)
            disp.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical', ax=ax)
            ax.set_title("Confusion Matrix on Test Data")
            plt.tight_layout()
            cm_plot_path = os.path.join(MODELS_DIR, "confusion_matrix_v3.png")
            fig.savefig(cm_plot_path)
            plt.close(fig)
            logger.info(f"Confusion Matrix plot saved to: {cm_plot_path}")
            mlflow.log_artifact(cm_plot_path)

            # METRICS
            p_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
            r_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Test F1-Micro: {f1_micro:.4f}")

            mlflow.log_metrics({
                "final_test_loss": history.history['val_loss'][-1],
                "final_test_accuracy": accuracy,
                "final_test_precision_micro": p_micro,
                "final_test_recall_micro": r_micro,
                "final_test_f1_micro": f1_micro
            })

            # SAVE CURVES
            plot_path = os.path.join(MODELS_DIR, "performance_metrics.png")
            save_training_curves(history, plot_path)
            mlflow.log_artifact(plot_path)

            # SAVE FINAL MODEL
            model.save(ckpt_path)
            logger.info("Model and all artifacts saved to %s", MODELS_DIR)
            logger.info("MLflow run completed successfully!")

    except Exception as e:
        logger.error("An error occurred: %s", e)


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    create_and_train_model()