from unittest.mock import Mock, patch

import pytest

from src.rag_service import RAGService


@pytest.fixture
def mock_rag_components():
    """
    Mock all external dependencies of RAGService:
    - FAISS vector store
    - ChatGroq LLM
    - ConversationalRetrievalChain
    """
    with patch("src.rag_service.FAISS.load_local") as mock_faiss:
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        mock_faiss.return_value = mock_vectorstore

        with patch("src.rag_service.ChatGroq") as mock_llm:
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            with patch(
                "src.rag_service.ConversationalRetrievalChain.from_llm"
            ) as mock_chain:
                mock_chain_instance = Mock()
                mock_chain_instance.invoke.return_value = {"answer": "Test answer"}
                mock_chain.return_value = mock_chain_instance

                yield  # Provide mocked environment for tests


def test_rag_service_init(mock_rag_components):
    """
    Test initialization of RAGService.
    Ensures that the vector store, retriever, LLM, and chain are properly set up.
    """
    service = RAGService(db_path="/fake/path")

    assert service.vectorstore is not None
    assert service.retriever is not None
    assert service.llm is not None
    assert service.chain is not None


def test_rag_chat(mock_rag_components):
    """
    Test the RAGService.chat method using a mocked retrieval and LLM chain.
    Verifies that a valid response dictionary with an answer and sources is returned.
    """
    service = RAGService(db_path="/fake/path")
    result = service.chat("Test query")

    assert "answer" in result
    assert result["answer"] == "Test answer"
    assert "sources" in result
