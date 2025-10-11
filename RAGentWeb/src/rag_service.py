import os
import logging
from pathlib import Path
from dotenv import load_dotenv
# Import Groq and its chat model
from langchain_groq import ChatGroq 
# Remove unused Ollama import
# from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain


logger = logging.getLogger("RAGService")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGService:
    """Manages the Retrieval-Augmented Generation (RAG) functionality."""

    def __init__(self, db_path: str):
        """Initializes the RAG components."""
        try:
            # 1. Load Embeddings (REMAINS THE SAME)
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Initializing HuggingFace Embeddings: {embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

            # 2. Load Vector Store (REMAINS THE SAME)
            self.vectorstore = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS Vector Store loaded from: {db_path}")

            # 3. Initialize Retriever and LLM (CHANGED TO GROQ)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # --- START GROQ INTEGRATION ---
            # Retrieve API Key from environment variable set in app.py
            groq_api_key = os.getenv("GROQ_API_KEY") 
            
            if not groq_api_key:
                 logger.error("FATAL: GROQ_API_KEY environment variable not found.")
                 raise ValueError("GROQ_API_KEY must be set to initialize RAG Service.")
                 
            # Use Groq's Chat model
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant",  # Fast and capable Groq model
                temperature=0.1
            )
            logger.info(f"LLM initialized: ChatGroq (llama3-8b-8192)")
            # --- END GROQ INTEGRATION ---

            self.chat_history = []

            # 4. Initialize the RAG Chain (REMAINS THE SAME)
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
            )

            logger.info(f"RAGService initialized successfully with Groq LLM.")

        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            # Reraise the exception so FastAPI startup fails gracefully
            raise e

    def chat(self, query: str) -> dict:
        """
        Processes a user query and returns an RAG-generated answer.
        """
        # ... (CHAT LOGIC REMAINS THE SAME) ...
        response = self.chain.invoke({"question": query, "chat_history": self.chat_history})

        self.chat_history.append((query, response["answer"]))
        if len(self.chat_history) > 5:
            self.chat_history = self.chat_history[-5:]

        return {
            "answer": response["answer"],
            # Updated source description
            "sources": "Data retrieved from local FAISS index (HuggingFace Embeddings) and answered by Groq LLM."
        }

# Example usage (Optional - keeps the self-test block functional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Simulate environment loading for local testing
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"
    
    try:
        rag_service = RAGService(db_path=str(FAISS_PATH))
        
        print("\n RAG Chat Start ")
        user_query_1 = "What are the common symptoms of the skin disease mentioned in the documents?"
        print(f"User: {user_query_1}")
        result_1 = rag_service.chat(user_query_1)
        print(f"RAG: {result_1['answer']}")
        print(f"Source: {result_1['sources']}")

    except Exception as e:
        print(f"\nFATAL ERROR: Could not run RAGService. Ensure the FAISS index is built. Error: {e}")