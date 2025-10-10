def map_symptoms(user_input, symptom_columns):
    """Map user input symptoms to binary vector."""
    symptom_vector = [0] * len(symptom_columns)
    for symptom in user_input.get('symptoms', []):
        if symptom in symptom_columns:
            symptom_vector[symptom_columns.index(symptom)] = 1
    return symptom_vector