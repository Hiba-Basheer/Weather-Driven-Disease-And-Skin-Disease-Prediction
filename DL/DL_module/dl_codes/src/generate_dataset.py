import pandas as pd
import random
import logging
from pathlib import Path
import re
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the weather-related disease dataset from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
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


def generate_text(row: pd.Series, symptom_columns: list, cities: list) -> str:
    """
    Generate descriptive text for each row in the dataset.

    Args:
        row (pd.Series): A row from the dataset.
        symptom_columns (list): List of symptom column names.
        cities (list): List of city names.

    Returns:
        str: Generated descriptive text.
    """
    symptoms = [col for col in symptom_columns if row.get(col, 0) == 1]
    symptom_str = ", ".join(symptoms) if symptoms else "no specific symptoms"
    gender = "male" if row.get("Gender", 0) == 1 else "female"
    age = row.get("Age", "unknown")
    city = random.choice(cities)

    return f"I am a {gender} aged {age}, living in {city}. I have been experiencing {symptom_str}."


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the modified dataset with generated text to a CSV file.
    Ensures 'prognosis' is the last column.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        output_path (str): Path to save the output CSV file.
    """
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
    """
    Creates a DataFrame of synthetic examples for 'Heart Attack', 'Stroke', and 'Migraine'.
    """
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
    Main function to load dataset, generate text, and save the updated dataset.
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
    combined_df['symptom_profile'] = combined_df.apply(
        lambda row: " ".join([col for col in symptom_cols if row[col] == 1]), axis=1
    )
    combined_df['text'] = combined_df.apply(generate_text, axis=1, args=(symptom_cols, cities))
    combined_df['symptom_count'] = combined_df[symptom_cols].sum(axis=1)

    save_dataset(combined_df, str(output_path))


if __name__ == "__main__":
    main()
