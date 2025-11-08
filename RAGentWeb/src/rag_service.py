"""
rag_service.py
Retrieval-Augmented Generation (RAG) service for health-related question answering.
Integrates:
  • FAISS for document retrieval
  • HuggingFace embeddings
  • Groq LLM for response generation
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("RAGService")

load_dotenv()


class RAGService:
    """
    Service class for Retrieval-Augmented Generation using Groq and FAISS.

    Steps:
        1. Load HuggingFace embeddings
        2. Load FAISS vector store
        3. Initialize retriever and Groq LLM
        4. Build ConversationalRetrievalChain for QA
    """

    def __init__(self, db_path: str):
        """Initializes RAG components with FAISS, embeddings, and Groq LLM."""
        try:
            # Embedding model
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Initializing HuggingFace embeddings: {embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

            # Load FAISS vector store
            self.vectorstore = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS vector store loaded from: {db_path}")

            # Retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            # LLM setup (Groq)
            groq_api_key = os.getenv("GROQ_API_KEY") or "dummy"
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not set in environment variables.")

            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
            )
            logger.info("Groq LLM initialized successfully.")

            # RAG chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, retriever=self.retriever
            )

            self.chat_history: list[tuple[str, str]] = []
            logger.info("RAGService initialized successfully.")

        except Exception as e:
            logger.error(f"RAGService initialization failed: {e}")
            raise e

    def chat(self, query: str) -> dict:
        """
        Processes a user query and returns an answer using RAG.

        Args:
            query (str): The user's input question.

        Returns:
            dict: A dictionary with the generated answer and source details.
        """
        try:
            response = self.chain.invoke(
                {"question": query, "chat_history": self.chat_history}
            )
            self.chat_history.append((query, response["answer"]))
            self.chat_history = self.chat_history[-5:]  # keep last 5 exchanges

            return {
                "answer": response["answer"],
                "sources": "Data retrieved from FAISS index using HuggingFace embeddings and Groq LLM.",
            }

        except Exception as e:
            logger.error(f"RAG chat error: {e}")
            return {"answer": "Error processing query.", "sources": str(e)}


# Evaluation
if __name__ == "__main__":
    """
    Evaluate RAGService on health-related questions.
    Measures:
        • Answer keyword relevance
        • Response time
    """
    import time

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    BASE_DIR = Path(__file__).resolve().parent.parent
    FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"

    try:
        service = RAGService(db_path=str(FAISS_PATH))
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit(1)

    test_cases = [
        (
            "What are the common symptoms of dengue fever?",
            ["fever", "headache", "joint pain", "rash"],
        ),
        (
            "What are the symptoms of malaria?",
            ["fever", "chills", "sweating", "headache", "fatigue"],
        ),
        (
            "What are the symptoms of heart attack?",
            ["chest pain", "shortness of breath", "nausea"],
        ),
        (
            "what should I do if I have eczema??",
            ["Consult a doctor", "Keep your skin moisturized", "Avoid triggers"],
        ),
    ]

    correct = 0
    total = len(test_cases)
    times = []

    print("\nRAG Service Evaluation")
    print("=" * 80)

    for i, (question, keywords) in enumerate(test_cases, 1):
        start = time.time()
        result = service.chat(question)
        elapsed = time.time() - start
        times.append(elapsed)

        answer = result["answer"].lower()
        matched = sum(1 for kw in keywords if kw.lower() in answer)

        print(f"\nTest {i}")
        print(f"Question: {question}")
        print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Keywords matched: {matched}/{len(keywords)} | Time: {elapsed:.2f}s")

        if matched == len(keywords):
            correct += 1
            print("Status: PASSED")
        else:
            print("Status: FAILED")

        print("-" * 80)

    accuracy = correct / total * 100
    avg_time = sum(times) / total

    print("\nFinal Report")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Average response time: {avg_time:.2f}s")
    print("=" * 80)
