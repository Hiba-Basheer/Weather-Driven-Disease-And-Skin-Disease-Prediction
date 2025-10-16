# ml


# import joblib

# # Paths to your ML files
# model_file = r"D:\brototype\week27\RAGentWeb\models\ml\trained_model.pkl"
# label_encoder_file = r"D:\brototype\week27\RAGentWeb\models\ml\label_encoder.pkl"
# features_file = r"D:\brototype\week27\RAGentWeb\models\ml\ml_expected_columns.pkl"

# # Load and inspect model
# model = joblib.load(model_file)
# print("Model type and params:", type(model), model.get_params())

# # Load and inspect label encoder
# label_encoder = joblib.load(label_encoder_file)
# print("Label encoder classes:", label_encoder.classes_)

# # Load and save expected features to text
# expected_features = joblib.load(features_file)
# print("Expected features:", expected_features)

# # Optional: save feature names to a file to copy easily
# with open("ml_expected_columns.txt", "w") as f:
#     for col in expected_features:
#         f.write(f"{col}\n")
# print("Saved feature names to ml_expected_columns.txt")




# dl

# import json

# with open(r"D:\brototype\week27\RAGentWeb\models\dl\text_vectorizer_config.json", "r", encoding="utf-8") as f:
#     config = json.load(f)

# print(json.dumps(config, indent=2))


# with open(r"D:\brototype\week27\RAGentWeb\models\dl\text_vectorizer_vocab.txt", "r", encoding="utf-8") as f:
#     vocab = [line.strip() for line in f if line.strip()]

# print(vocab)


# import joblib

# le = joblib.load(r"D:\brototype\week27\RAGentWeb\models\dl\label_encoder.pkl")
# print(le.classes_)

# import joblib






# import joblib

# scaler_path = r"D:\brototype\week27\RAGentWeb\models\dl\scaler_4_features.pkl"
# scaler = joblib.load(scaler_path)  # ✅ this is correct

# print("Scaler expects:", scaler.n_features_in_)
# print("Feature means:", scaler.mean_)
# print("Feature scales:", scaler.scale_)


# import os
# import asyncio
# import numpy as np
# from src.dl_service import DLService
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# MODEL_PATH = r"D:\brototype\week27\RAGentWeb\models\dl\dl_model.keras"
# API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")  # ✅ Reads from .env

# async def main():
#     service = DLService(MODEL_PATH, API_KEY)

#     test_texts = [
#         "I am 25 years old living in Calicut. I feel severe headache, fever, and nausea.",
#         "I am 40 years old living in Kochi. I feel chest pain and dizziness.",
#         "I am 30 years old living in Delhi. I feel cough, sore throat, and runny nose.",
#     ]

#     for text in test_texts:
#         print("\n🧠 Testing input:", text)
#         result = await service.predict(text)

#         # Show the top-3 probabilities (for debugging overconfidence)
#         if "raw_probs" in result:
#             probs = np.array(result["raw_probs"])
#             top3_idx = np.argsort(probs)[-3:][::-1]
#             print("Top 3 class probabilities:")
#             for i in top3_idx:
#                 print(f"  Class {i}: {probs[i]*100:.2f}%")

#         print("Predicted Disease:", result.get("prediction"))
#         print("Confidence:", f"{result.get('confidence', 0)*100:.2f}%")
#         print("Weather Data:", result.get("raw_weather_data", {}))
#         print("Status:", result.get("status"))
#         print("------")

# if __name__ == "__main__":
#     asyncio.run(main())








# import asyncio
# from .dl_service import DLService
# import os

# Load your API key from .env
# from dotenv import load_dotenv
# load_dotenv()
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# # Path to your trained DL model
# MODEL_PATH = os.path.join("models", "dl", "dl_model.keras")

# # Instantiate DL service
# dl_service = DLService(MODEL_PATH, OPENWEATHER_API_KEY)

# # Example multiple user inputs
# user_inputs = [
#     "I am 25 years old living in Kochi. I have severe headache and chest pain",
#     "I am 40 years old living in Calicut. Feeling fatigue, cough and fever",
#     "I am 30 years old living in Kochi. Severe dizziness and joint pain",
#     "I am 50 years old living in Calicut. Nausea, vomiting and high fever",
#     "I am 35 years old living in Kochi. Shortness of breath and coughing"
# ]

# async def test_predictions():
#     for text in user_inputs:
#         result = await dl_service.predict(text)
#         print("\nUser input:", text)
#         print("Prediction result:", result)

# # Run the async test
# asyncio.run(test_predictions())


# import os
# import asyncio
# from .dl_service import DLService

# # Load your DLService
# MODEL_PATH = r"D:models\dl\dl_model.keras"  # replace with your actual model path
# from dotenv import load_dotenv
# load_dotenv()
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
#    # or load from .env

# # Sample user inputs
# user_inputs = [
#     "I am 25 years old living in Kochi. I have severe headache and chest pain",
#     "I am 40 years old living in Calicut. Feeling fatigue, cough and fever",
#     "I am 30 years old living in Kochi. Severe dizziness and joint pain",
#     "I am 50 years old living in Calicut. Nausea, vomiting and high fever",
#     "I am 35 years old living in Kochi. Shortness of breath and coughing"
# ]

# async def main():
#     service = DLService(MODEL_PATH, OPENWEATHER_API_KEY)

#     for text in user_inputs:
#         result = await service.predict(text)
#         print("\n==============================")
#         print("User input:", text)
#         print("Prediction result:", result)
#         print("==============================\n")
        

# if __name__ == "__main__":
#     asyncio.run(main())




