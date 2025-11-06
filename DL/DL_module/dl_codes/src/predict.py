"""
predict.py
Take unstructured input (age, gender, city, symptoms) and predict disease using trained model.
"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers 
import joblib
import json
import logging
import requests 
from dotenv import load_dotenv 

# CONFIG 
BASE_DIR = r"D:\brototype\week27\DL"
MODELS_DIR = os.path.join(BASE_DIR, "DL_module", "dl_codes", "models")

# 1. LOAD API KEY FROM .ENV
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


#  MODEL CONSTANTS  
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
# Total features: 5 continuous + 65 symptoms + 1 gender = 71
TOTAL_CONTINUOUS_FEATURES = 5
TOTAL_BINARY_FEATURES = len(SYMPTOM_COLUMNS) + 1 # 66

#  PREPROCESSING FUNCTION 

def standardize_symptoms(text: str) -> str:
    """
    Standardizes common multi-word symptoms to the underscore format to match 
    the model's trained tokenization.
    """
    text = text.lower()
    
    # Pre-compiled list of multi-word symptoms to standardize
    standardization_map = {
        "throbbing headache": "throbbing_headache",
        "severe headache": "severe_headache",
        "pain behind the eyes": "pain_behind_the_eyes",
        "pain behind eyes": "pain_behind_eyes",
        "light sensitivity": "sensitivity_to_light",
        "sensitivity to light": "sensitivity_to_light",
        "sensitivity to sound": "sensitivity_to_sound",
        "blurred vision": "blurred_vision",
        "sudden numbness on one side": "sudden_numbness_on_one_side",
        "trouble speaking": "trouble_speaking",
        "loss of balance": "loss_of_balance",
        "weakness in arms or legs": "weakness_in_arms_or_legs",
        "reduced smell and taste": "reduced_smell_and_taste",
        "pain radiating to left arm": "pain_radiating_to_left_arm",
        "upper back pain": "upper_back_pain",
        "joint stiffness": "joint_stiffness",
        "rapid heart rate": "rapid_heart_rate",
        "rapid breathing": "rapid_breathing",
        "sinus headache": "sinus_headache",
        "shortness of breath": "shortness_of_breath",
        "sore throat": "sore_throat",
        "trouble seeing": "trouble_seeing",
        "swollen glands": "swollen_glands",
        "high fever": "high_fever",
        "abdominal pain": "abdominal_pain",
        "runny nose": "runny_nose",
        "facial pain": "facial_pain",
        "bleeding gums": "bleeding_gums",
        "loss of appetite": "loss_of_appetite",
        "skin irritation": "skin_irritation",
        "chest pain": "chest_pain",
        "hiv aids": "hiv_aids",
        "asthma history": "asthma_history",
        "jaw discomfort": "jaw_discomfort",
        "high blood pressure": "high_blood_pressure"
    }
    
    for old, new in standardization_map.items():
        text = text.replace(old, new)

    # Clean up punctuation and conjunctions
    text = re.sub(r'[,:\.\-\/]', ' ', text)
    text = re.sub(r'\band\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

#  CORE FUNCTIONS 

def fetch_weather(city_name: str):
    """Fetches real-time weather data using the OpenWeatherMap API."""
    if not OPENWEATHERMAP_API_KEY:
        logger.warning("OPENWEATHERMAP_API_KEY not found. Using default weather values.")
        # Default values: Temp=37C, Humidity=70%, Wind=5 m/s
        return 37.0, 70.0, 5.0 
        
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric" # Temp in Celsius, Wind in m/s
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        temp_c = data['main']['temp']
        humidity = data['main']['humidity']  # 0-100%
        wind_speed = data['wind']['speed'] # m/s
        
        logger.info(f"Fetched weather for {city_name}: Temp={temp_c}C, Humidity={humidity}%, Wind={wind_speed} m/s")
        return temp_c, humidity, wind_speed
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch weather for {city_name} (Error: {e}). Using defaults.")
        return 37.0, 70.0, 5.0 

def load_text_vectorizer(models_dir):
    """Loads the Keras TextVectorization layer from its saved config and vocabulary."""
    vectorizer_cfg_path = os.path.join(models_dir, "text_vectorizer_config.json")
    vocab_path = os.path.join(models_dir, "text_vectorizer_vocab.txt")
    
    with open(vectorizer_cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    
    vectorizer = layers.TextVectorization.from_config(config)
    vectorizer.set_vocabulary(vocab)
    logger.info("Successfully loaded TextVectorization layer from config/vocab.")
    return vectorizer


def load_assets():
    """Loads the trained model, scaler, label encoder, and text vectorizer."""
    try:
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dl_model.keras"))
        # Scaler is for 5 continuous features only
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        text_vectorizer = load_text_vectorizer(MODELS_DIR)
        return model, scaler, label_encoder, text_vectorizer
    except Exception as e:
        logger.error(f"Failed to load required assets from {MODELS_DIR}: {e}")
        raise


def preprocess_unstructured_input(raw_text: str, scaler, text_vectorizer):
    """
    Parses input text, fetches real-time weather, applies unit correction, extracts all 
    structured and text features, and converts input into model-ready tensors.
    
    Returns: X_num (5 features), X_struct_binary (66 features), X_text (text features)
    """
    
    # 1. Extract Patient Info (Age, Gender, City)
    age_match = re.search(r'age\s*(\d{1,3})', raw_text, re.IGNORECASE)
    age = float(age_match.group(1)) if age_match else 30.0
    
    gender_match = re.search(r'(?:gender|sex)\s*[:\-\s]*(\w+)|(male|female)', raw_text, re.IGNORECASE)
    gender_raw = None
    if gender_match:
        gender_raw = next(filter(None, gender_match.groups()), None) 

    # Gender binary encoding: 1 for Male, 0 for Female/other/default
    gender = 1 if (gender_raw and 'male' in gender_raw.lower()) else 0 
    
    city_match = re.search(r'(?:from|city|living in)\s*([a-zA-Z\s]+)', raw_text, re.IGNORECASE)
    city = city_match.group(1).strip() if city_match else None
    
    # 2. Determine Continuous Features (Weather & Count)
    if city:
        temperature, humidity, wind_speed = fetch_weather(city)
    else:
        logger.warning("City not found in input. Using default weather values.")
        temperature = 37.0
        humidity = 70.0
        wind_speed = 5.0 # m/s

    humidity_corrected = humidity / 100.0 
    wind_speed_corrected = wind_speed * 3.6
    
    # 3. Symptom Extraction 
    symptom_regex = r'(?:symptom[s]*[:\-\s]*|I\s+have\s+|I\'m\s+experiencing\s+|experiencing\s+|suffering\s+from\s+)(.*)'
    symptom_match = re.search(symptom_regex, raw_text, re.IGNORECASE)
    
    if symptom_match:
        symptom_profile_raw = symptom_match.group(1).strip()
    else:
        logger.warning("Symptom delimiter keyword ('symptoms:', 'I have', etc.) not found. Attempting to strip structured info.")
        symptom_profile_raw = re.sub(r'age\s*(\d{1,3})', '', raw_text, flags=re.IGNORECASE)
        symptom_profile_raw = re.sub(r'(?:gender|sex)\s*[:\-\s]*(\w+)|(male|female)', '', symptom_profile_raw, flags=re.IGNORECASE)
        symptom_profile_raw = re.sub(r'(?:from|city|living in)\s*([a-zA-Z\s]+)', '', symptom_profile_raw, flags=re.IGNORECASE)
        symptom_profile_raw = symptom_profile_raw.strip()


    # Standardize the symptoms for correct tokenization/feature extraction
    symptom_profile_standardized = standardize_symptoms(symptom_profile_raw)
    
    # Count words in the standardized profile for the continuous feature
    symptom_count = len(symptom_profile_standardized.split())
    
    # 4. Create 5 Continuous Features (Scaled) -> numeric_input
    numeric_features_raw = np.array([[age, temperature, humidity_corrected, wind_speed_corrected, symptom_count]])
    numeric_scaled = scaler.transform(numeric_features_raw) # Shape (1, 5)

    # 5. Create 66 Binary/Discrete Features (Gender + 65 Symptoms) -> structured_input
    binary_features = np.zeros((1, len(SYMPTOM_COLUMNS) + 1), dtype=np.float32)
    
    present_symptom_tokens = set(symptom_profile_standardized.split())
    
    # Set the 65 symptom features
    for i, col in enumerate(SYMPTOM_COLUMNS):
        if col in present_symptom_tokens:
            binary_features[0, i] = 1.0

    # Set the Gender feature 
    binary_features[0, -1] = float(gender)
    
    X_num = numeric_scaled        
    X_struct_binary = binary_features 
    
    # 6. Text Feature
    X_text = text_vectorizer([symptom_profile_standardized]).numpy() 

    # Return three separate arrays
    return X_num, X_struct_binary, X_text


def predict_disease(raw_text: str):
    """The main prediction pipeline."""
    
    try:
        # Assuming load_assets is correctly defined elsewhere
        model, scaler, label_encoder, text_vectorizer = load_assets()
    except Exception:
        return {"disease": "Error: Failed to load model assets. Check file paths.", "confidence": 0.0}

    # Get the three raw components: X_num (5), X_struct_binary (66), X_text (text)
    X_num, X_struct_binary, X_text = preprocess_unstructured_input(raw_text, scaler, text_vectorizer)
    X_struct_71 = np.hstack([X_num, X_struct_binary])
    
    pred_probs = model.predict({
        'numeric_input': X_struct_71, 
        'text_input': X_text
    }, verbose=0)
    
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(pred_probs))

    # Return the dictionary of results
    return {"disease": pred_label, "confidence": confidence}

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    print(" Running Multimodal Disease Prediction ")
    
    #  EVALUATION SESSION SETUP 
    evaluation_cases = [
        {
            "true_label": "Common Cold",
            "input_text": "Age 26, Gender: Female from Mumbai, symptoms: fever, headache, body pain"
        },
        {
            "true_label": "Common Cold",
            "input_text": "Age 55 male, city Delhi, I have sore throat and cough" 
        },
        {
            "true_label": "Migraine", 
            "input_text": "Age 40, Gender: Male from Delhi, symptoms: severe throbbing headache, pain behind the eyes, light sensitivity, and vomiting"
        },
        {
            "true_label": "Stroke", 
            "input_text": "Age 70, Gender: Female from San Francisco, I'm experiencing sudden numbness on one side, trouble speaking, blurred vision, loss of balance, and confusion"
        }
    ]

    correct_predictions = 0
    total_cases = len(evaluation_cases)
    
    print(f"\n STARTING EVALUATION OF {total_cases} TEST CASES ")
    
    for i, case in enumerate(evaluation_cases):
        input_text = case["input_text"]
        true_label = case["true_label"]
        
        # Run Prediction
        result = predict_disease(input_text)
        predicted_label = result["disease"]
        confidence = result["confidence"]
        
        # Determine Correctness
        is_correct = (predicted_label == true_label)
        
        if is_correct:
            correct_predictions += 1
        
        # Print Detailed Result
        status = "CORRECT" if is_correct else f"INCORRECT (True: {true_label})"
        
        print(f"\nCase {i+1}: {status}")
        print(f"  User Input: {input_text[:80]}...")
        print(f"  True Disease: {true_label}")
        print(f"  Prediction: {predicted_label} (Confidence: {confidence:.2f})")
        
    #  FINAL ACCURACY REPORT 
    accuracy = (correct_predictions / total_cases) * 100 if total_cases > 0 else 0
    
    print("\n" + "="*50)
    print("FINAL EVALUATION SUMMARY")
    print(f"Total Cases Tested: {total_cases}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)