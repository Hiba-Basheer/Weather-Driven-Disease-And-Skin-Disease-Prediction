import os
import logging
from pathlib import Path
from dotenv import load_dotenv
# LangChain Imports
from langchain_community.vectorstores import FAISS 
from langchain.docstore.document import Document 
from langchain.text_splitter import CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DATA_DIRS = [
    BASE_DIR / "data" / "skin_diseases",
    BASE_DIR / "data" / "weather_related_disease"
]
FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_SETUP")

# Document Loading 

def load_documents(data_dirs: list[Path]) -> list[Document]:
    """
    Loads all text-based documents from the specified RAG directories using manual file reading.
    Cleans and converts to LangChain Document objects.
    """
    logger.info(f"Starting document loading from: {[str(d) for d in data_dirs]}")
    all_documents = []
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}. Skipping.")
            continue
            
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                full_path = data_dir / file
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    # Create LangChain Document
                    doc = Document(page_content=content, metadata={"source": str(full_path)})
                    all_documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {full_path}: {e}")

    if not all_documents:
        logger.error("No documents were loaded. Check paths and file types.")
        return []
        
    logger.info(f"Loaded {len(all_documents)} documents.")
    return all_documents

# Text Splitting 

def split_text(documents: list[Document]) -> list[Document]:
    """
    Splits the loaded documents into smaller, manageable chunks using CharacterTextSplitter.
    (Matches chunk size/overlap from the second script).
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n" 
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks

# Vector DB Building 

def build_vector_db():
    """Builds and persists the FAISS vector database."""
    logger.info(f"Checking for existing DB at: {FAISS_PATH}") 
    
    # Check if the FAISS index directory already exists
    if FAISS_PATH.exists():
        # FAISS stores its index in a folder
        logger.info("Vector database (FAISS) already exists. Skipping build.")
        return

    documents = load_documents(RAG_DATA_DIRS)
    if not documents:
        logger.warning("Cannot build DB: No documents loaded.")
        return

    chunks = split_text(documents)

    # Initialize HuggingFace Embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Initializing HuggingFace Embeddings with model: {embedding_model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
        return


    logger.info("Creating FAISS database...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Ensure parent directory exists for saving
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True) 
    
    # Save the FAISS index locally
    vectordb.save_local(str(FAISS_PATH))
    logger.info(f"FAISS index build complete and saved to {FAISS_PATH}.")

if __name__ == "__main__":
    """Main execution block to run the preprocessing."""
    build_vector_db()
    logger.info("RAG data preprocessing finished.")