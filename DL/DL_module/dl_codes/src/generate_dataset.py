import pandas as pd
import random
import logging
from pathlib import Path
import os 
from groq import Groq 
import re
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GROQ_CLIENT = None
try:
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not GROQ_CLIENT.api_key:
        logger.error("GROQ_API_KEY not found. Text generation will use template fallback.")
        GROQ_CLIENT = None
    else:
        logger.info("Groq Client initialized successfully.")
except Exception as e:
    logger.error("Error during Groq client initialization: %s. Text generation will use template fallback.", e)
    GROQ_CLIENT = None

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the weather-related disease dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully from %s", file_path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

def create_llm_generation_prompt(row: pd.Series, symptom_columns: list, cities: list) -> str:
    """
    Generate a detailed prompt for the LLM based on the row's structured data.
    """
    symptoms = [col for col in symptom_columns if row.get(col, 0) == 1]
    symptom_str = ", ".join(symptoms) if symptoms else "no specific symptoms"
    
    gender_word = "male" if row.get("Gender", 0) == 1 else "female"
    age = row.get("Age", "unknown")
    city = random.choice(cities)
    temp = row.get("Temperature (C)", "N/A")
    humidity = row.get("Humidity", "N/A")
    prognosis = row.get("prognosis", "unknown")

    prompt = f"""
    You are a professional medical writer tasked with generating a realistic patient intake description. 
    Always respond with ONLY the patient description text, without any greetings, titles, or concluding remarks.
    
    **Context:**
    - Patient Age: {age}
    - Patient Gender: {gender_word}
    - Patient Location (for tone/background only): {city}
    - Weather: Temperature {temp}°C, Humidity {humidity}
    - Diagnosis (for context, DO NOT mention this in the output): {prognosis}
    
    **Core Symptoms to Describe:** {symptom_str}
    
    **Task:**
    Write a 1-3 sentence patient description reporting their symptoms.
    - CRUCIALLY, the language must be highly varied, informal, and sound like natural patient speech.
    - Do NOT use the phrasing 'I am a male/female aged X, living in Y...'
    - Do NOT just list the symptoms. Integrate them naturally.
    - Introduce temporal (time) or emotional language (e.g., 'since this morning', 'feel awful').
    - If symptoms are 'no specific symptoms', generate a neutral status update (e.g., 'I feel fine, just here for a check-up.').
    """
    return prompt

def generate_text_with_groq(prompt_text: str, fallback_text: str) -> str:
    """
    Calls the Groq API to generate text, with a fallback to the rule-based template.
    """
    if GROQ_CLIENT is None:
        return fallback_text

    try:
        system_message = {
            "role": "system",
            "content": "You are a professional medical writer tasked with generating a realistic, natural, and informal patient intake description based on provided facts. Always respond with only the patient description text, in 1-3 sentences."
        }
        
        user_message = {
            "role": "user",
            "content": prompt_text
        }
        
        # Using Llama 3 for fast, high-quality text generation
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=[system_message, user_message],
            model="llama3-8b-8192", 
            temperature=0.7, # Allows for more creative variability
            max_tokens=256
        )
        
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq API call failed: {e}. Falling back to rule-based text.")
        return fallback_text

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    try:
        if "prognosis" in df.columns:
            cols = [c for c in df.columns if c != "prognosis"] + ["prognosis"]
            df = df[cols]

        df.to_csv(output_path, index=False)
        logger.info("Generated dataset saved to %s", output_path)
    except Exception as e:
        logger.error("Error saving dataset: %s", e)
        raise

def create_synthetic_examples_df(symptom_cols: list, cities: list) -> pd.DataFrame:
    
    synthetic_data = []

    # Core symptoms for each disease
    heart_attack_core_symptoms = [
        'chest_pain', 'shortness_of_breath', 'sweating', 'nausea', 'dizziness', 'fatigue',
        'pain_radiating_to_left_arm', 'jaw_discomfort', 'upper_back_pain', 'rapid_heart_rate', 'anxiety'
    ]

    stroke_core_symptoms = [
        'sudden_numbness_on_one_side', 'trouble_speaking', 'severe_headache', 'dizziness',
        'loss_of_balance', 'blurred_vision', 'weakness_in_arms_or_legs', 'trouble_seeing',
        'slurred_speech', 'confusion'
    ]

    migraine_core_symptoms = [
        'severe_headache', 'sensitivity_to_light', 'sensitivity_to_sound', 'nausea',
        'vomiting', 'blurred_vision', 'dizziness', 'throbbing_headache', 'pain_behind_the_eyes'
    ]

    # Extra medically relevant symptoms
    extra_symptoms_map = {
        'Heart Attack': ['lightheadedness', 'jaw_pain', 'arm_pain'],
        'Stroke': ['slurred_speech', 'confusion', 'weakness'],
        'Migraine': ['pain_behind_the_eyes', 'throbbing_headache', 'nausea']
    }

    # Helper to generate synthetic rows
    def generate_rows(core_symptoms, disease, age_range, extra_symptoms_count=2):
        rows = []
        for _ in range(30):
            row_data = {col: 0 for col in symptom_cols}

            # Set core symptoms
            for symptom in core_symptoms:
                if symptom in symptom_cols:
                    row_data[symptom] = 1

            # Add extra medically relevant symptoms
            if disease in extra_symptoms_map:
                extra_symptoms = random.sample(extra_symptoms_map[disease], k=random.randint(0, extra_symptoms_count))
                for symptom in extra_symptoms:
                    if symptom in symptom_cols:
                        row_data[symptom] = 1

            # Add numeric data
            row_data['Age'] = random.randint(*age_range)
            row_data['Gender'] = random.choice([0, 1])
            row_data['Temperature (C)'] = round(random.uniform(36.0, 38.0), 3)
            row_data['Humidity'] = round(random.uniform(0.4, 0.9), 2)
            row_data['Wind Speed (km/h)'] = round(random.uniform(2.0, 20.0), 3)
            row_data['prognosis'] = disease

            rows.append(row_data)
        return rows

    # Generate synthetic data
    synthetic_data.extend(generate_rows(heart_attack_core_symptoms, 'Heart Attack', (45, 90)))
    synthetic_data.extend(generate_rows(stroke_core_symptoms, 'Stroke', (50, 95)))
    synthetic_data.extend(generate_rows(migraine_core_symptoms, 'Migraine', (18, 60)))

    synthetic_df = pd.DataFrame(synthetic_data)
    logger.info(f"Created {len(synthetic_df)} synthetic examples DataFrame.")
    return synthetic_df
    
    
def main():
    """
    Main function to load dataset, generate LLM text, and save the updated dataset.
    """
    input_path = Path("D:/brototype/week27/DL/DL_module/dl_codes/data/Weather-related disease prediction.csv")
    output_path = Path("D:/brototype/week27/DL/DL_module/dl_codes/data/generated_data.csv")

    symptom_cols = [
        'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills', 'fatigue', 'runny_nose',
        'pain_behind_the_eyes', 'dizziness', 'headache', 'chest_pain', 'vomiting', 'cough', 'shivering',
        'asthma_history', 'high_cholesterol', 'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps', 'asthma',
        'high_blood_pressure', 'severe_headache', 'weakness', 'trouble_seeing', 'fever', 'body_aches',
        'sore_throat', 'sneezing', 'diarrhea', 'rapid_breathing', 'rapid_heart_rate', 'pain_behind_eyes',
        'swollen_glands', 'rashes', 'sinus_headache', 'facial_pain', 'shortness_of_breath',
        'reduced_smell_and_taste', 'skin_irritation', 'itchiness', 'throbbing_headache', 'confusion',
        'back_pain', 'knee_ache', 'sweating', 'arm_pain', 'jaw_pain', 'lightheadedness', 'loss_of_appetite',
        'bleeding_gums', 'dry_skin', 'sensitivity_to_light', 'joint_stiffness', 'slurred_speech',
        'pain_radiating_to_left_arm', 'jaw_discomfort', 'upper_back_pain', 'anxiety', 'rapid_heart_rate',
        'sudden_numbness_on_one_side', 'trouble_speaking', 'blurred_vision', 'weakness_in_arms_or_legs',
        'loss_of_balance', 'sensitivity_to_sound'
    ]

    cities = [
        'Kochi', 'London', 'New York', 'Tokyo', 'Paris', 'Berlin',
        'Sydney', 'Toronto', 'Dubai', 'Mumbai', 'Bengaluru', 'Chicago', 'Lagos', 'Los Angeles',
        'San Francisco'
    ]
    
    # Load dataset
    weather_df = load_dataset(str(input_path))

    # Ensure all symptom columns exist 
    for symptom in symptom_cols:
        if symptom not in weather_df.columns:
            weather_df[symptom] = 0
            logger.info(f"Added missing column to original data: {symptom}")

    numeric_base_cols = ['Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
    for col in numeric_base_cols:
        if col not in weather_df.columns:
            weather_df[col] = 0
            logger.info(f"Added missing numeric base column to original data: {col}")

    # Generate synthetic dataset
    synthetic_df = create_synthetic_examples_df(symptom_cols, cities)

    # Combine datasets 
    all_combined_cols = list(set(weather_df.columns) | set(synthetic_df.columns))
    weather_df = weather_df.reindex(columns=all_combined_cols, fill_value=0)
    synthetic_df = synthetic_df.reindex(columns=all_combined_cols, fill_value=0)

    combined_df = pd.concat([weather_df, synthetic_df], ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")

    # Ensure symptom columns are numeric 
    for col in symptom_cols:
        combined_df[col] = pd.to_numeric(combined_df.get(col, 0), errors='coerce').fillna(0).astype(int)

    # Generate features 
    
    # 1. Create the symptom profile column 
    combined_df['symptom_profile'] = combined_df.apply(
        lambda row: " ".join([col for col in symptom_cols if row[col] == 1]), axis=1
    )
    
    def rule_based_text_generator(row: pd.Series):
        symptoms = row['symptom_profile'].replace(' ', ', ')
        gender = "male" if row.get("Gender", 0) == 1 else "female"
        age = row.get("Age", "unknown")
        city = random.choice(cities)
        return f"I am a {gender} aged {age}, living in {random.choice(cities)}. I have been experiencing {symptoms}."

    combined_df['rule_based_text_fallback'] = combined_df.apply(rule_based_text_generator, axis=1)

    # 3. Generate the new LLM-based text
    logger.info("Starting Groq-powered text augmentation...")
    
    llm_generated_texts = []
    
    for index, row in combined_df.iterrows():
        # Get the detailed prompt
        prompt = create_llm_generation_prompt(row, symptom_cols, cities)
        
        # Get the fallback text 
        fallback = row['rule_based_text_fallback']
        
        # Call Groq 
        llm_text = generate_text_with_groq(prompt, fallback)
        llm_generated_texts.append(llm_text)

    # Assign the new, realistic text to the 'text' column
    combined_df['text'] = llm_generated_texts
    
    # Drop the temporary fallback column
    combined_df = combined_df.drop(columns=['rule_based_text_fallback'])

    combined_df['symptom_count'] = combined_df[symptom_cols].sum(axis=1)

    save_dataset(combined_df, str(output_path))


if __name__ == "__main__":
    main()