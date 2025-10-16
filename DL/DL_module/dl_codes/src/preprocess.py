import pandas as pd
import numpy as np
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
import joblib
import json
import requests
import mlflow
import tensorflow as tf
from tensorflow.keras import layers 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load the .env file
dotenv_path = r"D:/brototype/week27/DL/.env"
load_dotenv(dotenv_path)



SYMPTOM_COLUMNS = [
    'chills', 'loss_of_balance', 'reduced_smell_and_taste', 'nasal_polyps', 'Humidity', 'pain_behind_eyes',
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



def fetch_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        temp_c = data['main']['temp']
        humidity = data['main']['humidity'] / 100  # convert % to 0-1
        wind_speed = data['wind']['speed']
        return temp_c, humidity, wind_speed
    else:
        print("Failed to fetch weather, using defaults")
        return 25.0, 0.70, 10.0


def preprocess_user_input(user_input: str, scaler, text_vectorizer):
    """
    Preprocesses a single user input string into the format expected by the model.
    """
    if not isinstance(user_input, str):
        raise TypeError("Input must be a string.")

    # Extract numeric data
    age_match = re.search(r'age(?:d)?\s*(\d+)', user_input, re.IGNORECASE)
    age = float(age_match.group(1)) if age_match else np.nan

    gender_match = re.search(r'\b(male|female)\b', user_input, re.IGNORECASE)
    gender = 1.0 if gender_match and gender_match.group(1).lower() == 'male' else 0.0

    # Extract symptoms 
    symptom_patterns = [
        r'experiencing\s*([^.]+)',
        r'symptoms\s*[:\-]?\s*([^.]+)',
        r'have\s+([a-zA-Z_,\s]+)',
        r'feeling\s+([a-zA-Z_,\s]+)'
    ]
    
    symptoms_text = ""
    for pattern in symptom_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            symptoms_text = match.group(1)
            break

    # Extract known symptom keywords directly
    if not symptoms_text:
        symptoms_text = " ".join([
            s.replace('_', ' ') for s in SYMPTOM_COLUMNS if re.search(r'\b' + re.escape(s.replace('_', ' ')) + r'\b', user_input, re.IGNORECASE)
        ])

    # Split symptoms and clean them up
    symptoms_list_raw = [
        s.strip().replace('.', '').replace(',', '') for s in re.split(r'[,;]\s*|and\s*', symptoms_text)
    ]

    symptoms_list_processed = []
    for s in symptoms_list_raw:
        cleaned_s = s.lower().replace(' ', '_')
        
        # Explicit mappings for common variations or similar concepts
        if cleaned_s == 'pain_in_left_arm' and 'pain_radiating_to_left_arm' in SYMPTOM_COLUMNS:
            symptoms_list_processed.append('pain_radiating_to_left_arm')
        elif cleaned_s == 'low_grade_fever' and 'fever' in SYMPTOM_COLUMNS:
            symptoms_list_processed.append('fever')
        elif cleaned_s == 'mild_headache' and 'headache' in SYMPTOM_COLUMNS:
            symptoms_list_processed.append('headache')
        elif cleaned_s == 'weakness_in_arms_or_legs' and 'weakness' in SYMPTOM_COLUMNS:
             symptoms_list_processed.append('weakness')
        elif cleaned_s == 'trouble_speaking' and 'slurred_speech' in SYMPTOM_COLUMNS:
             symptoms_list_processed.append('slurred_speech')
        elif cleaned_s in SYMPTOM_COLUMNS:
            symptoms_list_processed.append(cleaned_s)

    symptoms_list = [s for s in symptoms_list_processed if s]

    # Recognized vs unrecognized
    recognized_symptoms = []
    unrecognized_symptoms = []
    vocab = set(text_vectorizer.get_vocabulary())

    for symptom in symptoms_list:
        if symptom in vocab:
            recognized_symptoms.append(symptom)
        else:
            unrecognized_symptoms.append(symptom)

    logger.info(f"Recognized Symptoms: {recognized_symptoms}")
    logger.info(f"Unrecognized Symptoms: {unrecognized_symptoms}")

    # One-hot encode symptoms
    symptoms_df = pd.DataFrame(columns=SYMPTOM_COLUMNS, index=[0]).fillna(0)
    for symptom in symptoms_list:
        if symptom in symptoms_df.columns:
            symptoms_df.loc[0, symptom] = 1

    symptom_profile = " ".join(symptoms_list)
    symptom_count = sum(symptoms_df.loc[0, SYMPTOM_COLUMNS])

    # Extract city from user input
    city_match = re.search(r'living in ([a-zA-Z\s]+)', user_input, re.IGNORECASE)
    city = city_match.group(1).strip() if city_match else None
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise ValueError("Weather API key not found in .env file")

    if city:
        temp_c, humidity, wind_speed = fetch_weather(city, api_key)
    else:
        temp_c, humidity, wind_speed = 25.0, 0.70, 10.0
    numeric_df = pd.DataFrame({
        'Age': [age],
        'Temperature (C)': [temp_c],
        'Humidity': [humidity],
        'Wind Speed (km/h)': [wind_speed],
        'symptom_count': [symptom_count]
    })

    X_numeric = scaler.transform(numeric_df.values)
    X_text = text_vectorizer(np.array([symptom_profile])).numpy()

    return X_numeric, X_text, symptom_profile


# Main execution for testing
if __name__ == "__main__":
    # Load preprocessors 
    try:
        scaler = joblib.load('artifacts/scaler.pkl')
        vectorizer_config_path = 'artifacts/text_vectorizer_config.json'
        with open(vectorizer_config_path, 'r') as f:
            vectorizer_config = json.load(f)
        vocab_path = 'artifacts/text_vectorizer_vocab.txt'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        vectorizer = layers.TextVectorization.from_config(vectorizer_config)
        vectorizer.set_vocabulary(vocab)
        
        user_input_example = 'I am a female aged 29, living in San Francisco. I have been experiencing severe_headache, sensitivity_to_light, sensitivity_to_sound, nausea, vomiting, blurred_vision, and dizziness.'
        
        X_numeric_ex, X_text_ex, _ = preprocess_user_input(user_input_example, scaler, vectorizer)
        print(f"Preprocessed Numeric Data:\n{X_numeric_ex}")
        print(f"Preprocessed Text Data:\n{X_text_ex}")
        
    except FileNotFoundError:
        print("Please run train.py first to create the preprocessor files in the 'artifacts' directory.")