"""
Main chatbot script integrating ML, DL, RAG, and image classification modules.
It supports:
- ML predictions using a scikit-learn model.
- DL/RAG-based question answering with LangChain.
- Skin image classification using a ResNet model.
- Data enrichment with weather information.
- Logging user inputs and system responses to CSV.

"""

import os
import re
import logging
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.image_module import predict_from_image
from src.weather_utils import extract_city, fetch_weather
from src.preprocess import preprocess_ml_input, preprocess_dl_text
import joblib
import hashlib

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found in .env file. RAG will not work.")
if not OPENWEATHER_API_KEY:
    logging.warning("OPENWEATHER_API_KEY not found. Weather enrichment will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
ML_MODEL_PATH = "models/trained_model.pkl"
DL_MODEL_PATH = "models/dl_model.keras"
RESNET_MODEL_PATH = "models/resnet_model.h5"
VECTOR_STORE_PATH = "data/vector_store/faiss_index"
USER_DATA_LOG = os.path.join(os.getcwd(), "data", "user_inputs.csv")

# Load models
try:
    ml_model = joblib.load(ML_MODEL_PATH)
    logging.info("ML model loaded successfully.")
except Exception as e:
    ml_model = None
    logging.warning(f"Failed to load ML model: {e}")

try:
    dl_model = tf.keras.models.load_model(DL_MODEL_PATH)
    logging.info("DL model loaded successfully.")
except Exception as e:
    dl_model = None
    logging.warning(f"Failed to load DL model: {e}")

try:
    resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
    logging.info("ResNet model loaded successfully.")
except Exception as e:
    resnet_model = None
    logging.warning(f"Failed to load ResNet model: {e}")


def create_qa_chain():
    """
    Create and initialize a Retrieval-Augmented Generation (RAG) pipeline.

    Loads HuggingFace embeddings, FAISS vector store, and OpenAI LLM.
    Returns:
        RetrievalQA object if successful, else None.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logging.info("Embeddings loaded.")

        vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        logging.info("FAISS vector store loaded.")

        retriever = vectordb.as_retriever()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
        logging.info("OpenAI LLM initialized.")

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        logging.info("RAG pipeline initialized successfully.")
        return qa_chain
    except Exception as e:
        logging.error(f"Failed to create QA chain: {e}")
        return None


qa_chain = create_qa_chain()
if not qa_chain:
    logging.error("RAG pipeline failed to initialize. Check FAISS index and OpenAI key.")
    logging.warning("qa_chain is None. RAG pipeline may not respond.")


def detect_module(query: str):
    """
    Detect which module should handle the user query.

    Args:
        query (str): User input query.

    Returns:
        tuple: (module_name, normalized_input)
            module_name can be 'skin_image', 'skin_image_with_text', 'ml', or 'dl'.
    """
    normalized_query = query.replace("\\", "/").strip().strip('"')
    query_lower = normalized_query.lower()

    if query_lower.endswith((".jpg", ".jpeg", ".png")) and os.path.exists(normalized_query):
        return "skin_image", normalized_query

    if "image:" in query_lower and "note:" in query_lower:
        return "skin_image_with_text", normalized_query

    if "age:" in query_lower and "gender:" in query_lower and "symptom:" in query_lower:
        return "ml", normalized_query

    return "dl", normalized_query


def handle_ml_input(user_input: str):
    """
    Handle user input for ML-based prediction.

    Args:
        user_input (str): Structured input containing features like age, gender, symptoms, city.

    Returns:
        str: Prediction result with explanation or error message.
    """
    features = {}
    for line in user_input.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            features[key.strip().lower()] = val.strip()

    logging.info(f"Extracted ML features: {features}")

    city = features.get("location") or features.get("city") or extract_city(user_input)
    weather = fetch_weather(city) if city else {}

    if ml_model:
        try:
            processed = preprocess_ml_input(features)
            prediction = ml_model.predict(processed)
            predicted_label = prediction[0]
            logging.info(f"ML prediction: {predicted_label}")

            explanation_prompt = (
                f"The ML model predicted '{predicted_label}' based on the following features:\n"
                f"{features}\nCan you explain what this condition means and what the user might want to know?"
            )

            rag_response = qa_chain.run(explanation_prompt) if qa_chain else "No RAG explanation available."
            result = f"ML Prediction: {predicted_label}\nExplanation: {rag_response}"
            save_user_data("ml", user_input, result, weather)
            return result
        except Exception as e:
            logging.error(f"Error during ML prediction: {e}")
            return f"Error in ML prediction: {e}"
    return "ML model not available."


def handle_dl_input(user_input: str):
    """
    Handle user input for DL/RAG pipeline.

    Args:
        user_input (str): Free-form user query.

    Returns:
        str: RAG pipeline response enriched with weather info if available.
    """
    city = extract_city(user_input)
    weather = fetch_weather(city) if city else {}

    if qa_chain:
        try:
            result = qa_chain.run(user_input)
            logging.info("RAG response generated.")
            enriched = (
                f"City: {city}\nWeather: {weather}\n\nRAG Answer: {result}"
                if city and weather else f"RAG Answer: {result}"
            )
            save_user_data("dl", user_input, enriched, weather)
            return enriched
        except Exception as e:
            logging.error(f"Error in RAG pipeline: {e}")
            return f"Error in RAG pipeline: {e}"
    logging.warning("qa_chain is not available during DL input handling.")
    return "RAG pipeline not available."


def handle_skin_image_input(image_path: str):
    """
    Handle skin image input for classification.

    Args:
        image_path (str): Path to the skin image file.

    Returns:
        str: Prediction result with confidence score or error message.
    """
    image_path = image_path.replace("\\", "/").strip().strip('"')

    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return f"Image not found: {image_path}"

    try:
        predicted_class, confidence = predict_from_image(image_path)
        logging.info(f"Image prediction: {predicted_class} ({confidence:.2f})")
        result = f"Skin Image Prediction: {predicted_class} (Confidence: {confidence:.2f})"
        save_user_data("skin_image", image_path, result)
        return result
    except Exception as e:
        logging.error(f"Error in image classification: {e}")
        return f"Error in image classification: {e}"


def handle_skin_image_with_text(image_path: str, user_note: str):
    """
    Handle skin image input with an accompanying user note.

    Args:
        image_path (str): Path to the skin image.
        user_note (str): Text description provided by the user.

    Returns:
        str: Combined prediction and RAG explanation.
    """
    image_path = image_path.replace("\\", "/").strip().strip('"')

    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return f"Image not found: {image_path}"

    try:
        predicted_class, confidence = predict_from_image(image_path)
        rag_response = qa_chain.run(user_note) if qa_chain else "No RAG response available."

        result = (
            f"Skin Image Prediction: {predicted_class} (Confidence: {confidence:.2f})\n"
            f"Based on your note: {rag_response}"
        )
        save_user_data("skin_image_with_text", f"{image_path} + {user_note}", result)
        return result
    except Exception as e:
        logging.error(f"Error in combined image-text handling: {e}")
        return f"Error processing image and note: {e}"

def generate_record_hash(record: dict) -> str:
    """Generate a hash from the record to detect duplicates."""
    record_str = f"{record['module']}-{record['user_input']}-{record['response']}"
    return hashlib.md5(record_str.encode()).hexdigest()

def save_user_data(module: str, user_input: str, response: str, weather: dict = None):
    """
    Save user query, system response, and additional metadata into CSV log.

    Args:
        module (str): Module name (ml, dl, skin_image, etc.).
        user_input (str): User input text.
        response (str): Chatbot response.
        weather (dict): Weather information dictionary.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        age_match = re.search(r"\b(?:i am|age[:\s]*)\s*(\d{1,3})\b", user_input.lower())
        age = age_match.group(1) if age_match else ""
        city = extract_city(user_input)

        record = {
            "timestamp": timestamp,
            "module": module,
            "user_input": user_input,
            "response": response,
            "age": age,
            "city": city if city else "",
            "temperature": weather.get("temperature") if weather else "",
            "humidity": weather.get("humidity") if weather else "",
            "condition": weather.get("condition") if weather else "",
            "wind_speed": weather.get("wind_speed") if weather else ""
        }
        
        # Generate hash for current record
        record_hash = generate_record_hash(record)

        # Check for existing hashes
        if os.path.exists(USER_DATA_LOG):
            try:
                existing_df = pd.read_csv(USER_DATA_LOG)
                existing_hashes = existing_df.apply(lambda row: generate_record_hash(row.to_dict()), axis=1)
                if record_hash in existing_hashes.values:
                    logging.info("Duplicate record detected. Skipping save.")
                    return
            except Exception as e:
                logging.warning(f"Could not check for duplicates: {e}")

        logging.info(f"Saving record: {record}")
        write_header = not os.path.isfile(USER_DATA_LOG) or os.path.getsize(USER_DATA_LOG) == 0
        df = pd.DataFrame([record])
        df.to_csv(USER_DATA_LOG, mode="a", header=write_header, index=False, encoding="utf-8")
        logging.info("User data saved to CSV.")
    except Exception as e:
        logging.error(f"Failed to save user data: {e}")


def chat():
    """
    Start chatbot interactive session.

    Continuously prompts user for input and routes to appropriate handler.
    Type 'exit' to quit the chatbot.
    """
    print("Chatbot ready. Type your input or 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break

        module, normalized_input = detect_module(user_input)
        logging.info(f"Detected module: {module}")

        if module == "ml":
            response = handle_ml_input(normalized_input)
        elif module == "dl":
            response = handle_dl_input(normalized_input)
        elif module == "skin_image":
            response = handle_skin_image_input(normalized_input)
        elif module == "skin_image_with_text":
            try:
                image_path = input("Enter image path: ").strip()
                user_note = input("Describe your condition: ").strip()
                response = handle_skin_image_with_text(image_path, user_note)
            except Exception as e:
                logging.error(f"Error handling image + text input: {e}")
                response = f"Error handling image + text input: {e}"
        else:
            response = "Could not determine the appropriate module for your input."

        print(f"\nBot: {response}")


if __name__ == "__main__":
    """
    Entry point of the script.

    Optionally performs a RAG pipeline test with a dengue fever query,
    then starts chatbot session.
    """
    if qa_chain:
        test_response = qa_chain.run("What is dengue fever?")
        print("RAG test response:", test_response)
    else:
        print("RAG pipeline is not available.")

    chat()
