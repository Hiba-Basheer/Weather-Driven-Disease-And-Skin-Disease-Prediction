"""
preprocess_and_train.py

This script handles all data preparation (loading, cleaning, scaling, encoding) 
and defines the multimodal deep learning model architecture used for training.

It saves all necessary preprocessing artifacts (scaler, vectorizer, encoder) 
for later use in the inference stage.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths 
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Model Hyperparameters 
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 128 
DENSE_UNITS_NUMERIC_1 = 64  
DENSE_UNITS_NUMERIC_2 = 32  
DROPOUT_RATE_NUMERIC = 0.3  
DROPOUT_RATE_TEXT = 0.3     
DROPOUT_RATE_FINAL = 0.3    
EPOCHS = 50 
BATCH_SIZE = 32

# Columns definition
CONTINUOUS_COLS = ['Age', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'symptom_count']


# Core Preprocessing Logic

def preprocess_data(df: pd.DataFrame, symptom_columns: list, data_path: Path) -> tuple:
    """
    Performs multimodal preprocessing: Label Encoding, Standard Scaling, and Text Vectorization.
    Saves preprocessors and returns processed data and model parameters.
    """
    logger.info(f"Starting multimodal data preprocessing for {data_path.name}")
    
    # 1. Target Encoding
    label_encoder = LabelEncoder()
    df['prognosis_encoded'] = label_encoder.fit_transform(df['prognosis'])
    num_classes = len(label_encoder.classes_)
    y_labels = df['prognosis_encoded'].values
    
    # Compute class weights for imbalance mitigation 
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_labels),
        y=y_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Calculated Class Weights: {class_weight_dict}")

    # 2. Structured Data Preparation (Scaling)
    
    # Identify binary/one-hot symptom columns + Gender
    binary_cols = [col for col in symptom_columns if col in df.columns and col not in CONTINUOUS_COLS]
    binary_cols.append('Gender') 
    
    # Standard Scaling for Continuous Features
    scaler = StandardScaler()
    X_continuous_scaled = scaler.fit_transform(df[CONTINUOUS_COLS])
    X_binary = df[binary_cols].values
    
    # Combine structured features
    X_structured = np.hstack([X_continuous_scaled, X_binary])
    logger.info(f"Structured Input Shape: {X_structured.shape} ({len(CONTINUOUS_COLS)} continuous + {len(binary_cols)} binary)")

    # 3. Unstructured Data Preparation (Text Vectorization)
    text_data = df['text'].astype(str).values
    
    text_vectorizer = layers.TextVectorization(
        max_tokens=None, 
        output_mode='int',
        output_sequence_length=MAX_SEQ_LENGTH
    )
    text_vectorizer.adapt(text_data)
    
    X_unstructured = text_vectorizer(text_data).numpy()
    vocab_size = text_vectorizer.vocabulary_size()

    # Save Preprocessing Artifacts
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.pkl')
    logger.info("Saved StandardScaler to artifacts/scaler.pkl")
    
    vectorizer_config = text_vectorizer.get_config()
    with open(ARTIFACTS_DIR / 'text_vectorizer_config.json', 'w') as f:
        json.dump(vectorizer_config, f)
    
    vocab = text_vectorizer.get_vocabulary()
    with open(ARTIFACTS_DIR / 'text_vectorizer_vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab) + '\n')
    logger.info(f"Saved TextVectorizer artifacts (Vocab Size: {vocab_size}).")
    
    joblib.dump(label_encoder, ARTIFACTS_DIR / 'label_encoder.pkl')
    logger.info("Saved LabelEncoder to artifacts/label_encoder.pkl")

    return X_structured, X_unstructured, y_labels, num_classes, vocab_size, class_weight_dict


# Multimodal Model Definition

def create_multimodal_model(
    structured_input_shape: int,
    vocab_size: int, 
    num_classes: int,
    max_seq_length: int = MAX_SEQ_LENGTH, 
    embedding_dim: int = EMBEDDING_DIM, 
    lstm_units: int = LSTM_UNITS,    
) -> Model:
    """
    Creates a multimodal DL model with separate branches for structured (numeric) 
    and text data, designed to balance feature influence. 
    """
    
    # 1. Structured (Numeric) Input Branch
    structured_input = tf.keras.Input(shape=(structured_input_shape,), name='numeric_input')
    
    x_num = layers.Dense(DENSE_UNITS_NUMERIC_1, activation='relu')(structured_input) 
    x_num = layers.Dropout(DROPOUT_RATE_NUMERIC)(x_num)                             
    
    # Added second layer for depth
    x_num = layers.Dense(DENSE_UNITS_NUMERIC_2, activation='relu')(x_num)          
    x_num = layers.Dropout(DROPOUT_RATE_NUMERIC)(x_num)                            
    
    # 2. Text Input Branch
    text_input = tf.keras.Input(shape=(max_seq_length,), name='text_input')
    
    # Text Embedding and LSTM with higher capacity
    x_text = layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        mask_zero=True
    )(text_input)
    x_text = layers.Bidirectional(layers.LSTM(lstm_units))(x_text)
    x_text = layers.Dropout(DROPOUT_RATE_TEXT)(x_text) 

    # 3. Concatenate and Output
    combined = layers.concatenate([x_num, x_text])
    
    # Final classification layer
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(DROPOUT_RATE_FINAL)(z) 
    z = layers.Dense(num_classes, activation='softmax')(z)
    
    model = models.Model(inputs=[structured_input, text_input], outputs=z)
    
    return model


def train_model_runner(data_path: Path, symptom_columns: list) -> dict:
    """
    Loads data, preprocesses, splits data, and prepares the model for training.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please ensure it exists.")
        return {}

    # Preprocess the entire dataset
    X_structured, X_unstructured, Y_labels, NUM_CLASSES, VOCAB_SIZE, class_weight_dict = preprocess_data(df, symptom_columns, data_path)
    
    # Split the data into training and testing sets (Stratified)
    logger.info("Splitting data into train and test sets (stratified)")
    X_struct_train, X_struct_test, X_unstruct_train, X_unstruct_test, Y_train, Y_test = train_test_split(
        X_structured, X_unstructured, Y_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=Y_labels
    )
    
    # Determine the input shape for the structured branch
    structured_input_shape = X_struct_train.shape[1]
    
    # Create the model
    model = create_multimodal_model(
        structured_input_shape=structured_input_shape, 
        vocab_size=VOCAB_SIZE, 
        num_classes=NUM_CLASSES
    )
    model.summary(print_fn=logger.info)

    # Define Early Stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,          
        restore_best_weights=True
    )

    logger.info("The model and data are prepared. Call model.compile() and model.fit() in your separate training script.")
    
    return {
        'model': model,
        'X_train': {'numeric_input': X_struct_train, 'text_input': X_unstruct_train},
        'Y_train': Y_train,
        'X_test': {'numeric_input': X_struct_test, 'text_input': X_unstruct_test},
        'Y_test': Y_test,
        'class_weights': class_weight_dict,
        'callbacks': [early_stopping],
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS
    }