"""
Preprocessing and embedding pipeline for RAG chatbot.
Loads disease fact sheets, cleans and chunks text, and builds FAISS index.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_DIR = "data"
SOURCE_FOLDERS = [
    "skin_diseases",
    "weather_related_disease"
]
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "faiss_index")

def load_documents(folder_path: str) -> list:
    """
    Loads and cleans all .txt files from a folder.

    Parameters:
        folder_path (str): Path to folder containing .txt files.

    Returns:
        list: List of LangChain Document objects.
    """
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            full_path = os.path.join(folder_path, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                doc = Document(page_content=content, metadata={"source": full_path})
                documents.append(doc)
            except Exception as e:
                logging.warning(f"Failed to load {full_path}: {e}")
    logging.info(f"Loaded {len(documents)} documents from {folder_path}")
    return documents

def chunk_documents(documents: list) -> list:
    """
    Splits documents into smaller chunks for embedding.

    Parameters:
        documents (list): List of LangChain Document objects.

    Returns:
        list: Chunked documents.
    """
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logging.info(f"Chunked into {len(chunks)} segments.")
    return chunks

def build_vector_store():
    """
    Builds FAISS vector store from disease documents.
    """
    all_docs = []
    for folder in SOURCE_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(folder_path):
            docs = load_documents(folder_path)
            all_docs.extend(docs)
        else:
            logging.warning(f"Folder not found: {folder_path}")

    if not all_docs:
        logging.error("No documents found. Aborting vector store build.")
        return

    chunks = chunk_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTOR_STORE_PATH)
    logging.info(f"FAISS index saved to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    build_vector_store()