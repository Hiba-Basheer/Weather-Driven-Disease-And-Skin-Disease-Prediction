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

