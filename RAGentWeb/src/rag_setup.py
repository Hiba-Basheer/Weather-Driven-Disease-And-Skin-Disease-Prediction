"""
rag_vector_builder.py
Prepares and builds the FAISS vector database for RAGService.
Loads text documents, splits them into chunks, embeds them using
HuggingFace sentence transformers, and stores them in FAISS format.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Environment and paths
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DATA_DIRS = [
    BASE_DIR / "data" / "skin_diseases",
    BASE_DIR / "data" / "weather_related_disease",
]
FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGVectorBuilder")

# Document loading
def load_documents(data_dirs: list[Path]) -> list[Document]:
    """
    Loads all .txt files from the given directories and returns LangChain Document objects.
    """
    logger.info(f"Loading documents from: {[str(d) for d in data_dirs]}")
    all_documents = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}. Skipping.")
            continue

        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                file_path = data_dir / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        all_documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

    if not all_documents:
        logger.error("No valid documents loaded. Check your data directories.")
    else:
        logger.info(f"Loaded {len(all_documents)} documents successfully.")
    return all_documents

# Text splitting
def split_text(documents: list[Document]) -> list[Document]:
    """
    Splits long documents into smaller chunks for vectorization.
    """
    if not documents:
        logger.warning("No documents to split.")
        return []

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} text chunks.")
    return chunks

# Vector DB building
def build_vector_db():
    """
    Builds and saves the FAISS vector database using HuggingFace embeddings.
    """
    logger.info(f"Checking for existing FAISS index at: {FAISS_PATH}")
    if FAISS_PATH.exists():
        logger.info("FAISS index already exists. Skipping rebuild.")
        return

    documents = load_documents(RAG_DATA_DIRS)
    if not documents:
        logger.error("No documents found. Vector DB build aborted.")
        return

    chunks = split_text(documents)
    if not chunks:
        logger.error("No text chunks generated. Vector DB build aborted.")
        return

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Initializing embeddings model: {embedding_model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return

    logger.info("Building FAISS vector database...")
    vectordb = FAISS.from_documents(chunks, embeddings)

    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(FAISS_PATH))
    logger.info(f"FAISS index built and saved successfully at {FAISS_PATH}")

# Entry point
if __name__ == "__main__":
    build_vector_db()
    logger.info("RAG data preprocessing completed successfully.")
