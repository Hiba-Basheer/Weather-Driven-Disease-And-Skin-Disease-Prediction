import os
import logging
from pathlib import Path
from langchain_community.llms import Ollama  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain


logger = logging.getLogger("RAGService")

class RAGService:
    """Manages the Retrieval-Augmented Generation (RAG) functionality."""

    def __init__(self, db_path: str):
        """Initializes the RAG components."""
        try:
            # 1. Load Embeddings
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Initializing HuggingFace Embeddings: {embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

            # 2. Load Vector Store
            self.vectorstore = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS Vector Store loaded from: {db_path}")

            # 3. Initialize Retriever and LLM
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # This connects to the local Ollama server 
            self.llm = Ollama(model="phi3:mini", temperature=0.1)

            self.chat_history = []

            # 4. Initialize the RAG Chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
            )

            logger.info(f"RAGService initialized successfully with FAISS DB at: {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            raise e

    def chat(self, query: str) -> dict:
        """
        Processes a user query and returns an RAG-generated answer.
        """
        response = self.chain.invoke({"question": query, "chat_history": self.chat_history})

        self.chat_history.append((query, response["answer"]))
        if len(self.chat_history) > 5:
            self.chat_history = self.chat_history[-5:]

        return {
            "answer": response["answer"],
            "sources": "Data retrieved from local FAISS index (HuggingFace Embeddings)."
        }

# Example usage 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"
    
    try:
        rag_service = RAGService(db_path=str(FAISS_PATH))
        
        # Initial query
        print("\n RAG Chat Start ")
        user_query_1 = "What are the common symptoms of the skin disease mentioned in the documents?"
        print(f"User: {user_query_1}")
        result_1 = rag_service.chat(user_query_1)
        print(f"RAG: {result_1['answer']}")
        print(f"Source: {result_1['sources']}")

    except Exception as e:
        print(f"\nFATAL ERROR: Could not run RAGService. Ensure the FAISS index is built. Error: {e}")