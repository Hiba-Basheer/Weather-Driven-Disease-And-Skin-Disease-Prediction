# RAGentWeb/tests/test_rag_service.py
from unittest.mock import MagicMock, patch
from src.rag_service import RAGService


@patch("os.getenv", return_value="fake_key")
@patch("src.rag_service.FAISS.load_local", return_value=MagicMock())
@patch("src.rag_service.ConversationalRetrievalChain")
def test_rag_chat(mock_chain_class, mock_faiss, mock_getenv):
    # Mock the chain
    mock_chain_instance = MagicMock()
    mock_chain_instance.invoke.return_value = {
        "answer": "Test answer",
        "source_documents": [
            MagicMock(metadata={"source": "doc1"}),
            MagicMock(metadata={"source": "doc2"})
        ]
    }
    mock_chain_class.from_llm.return_value = mock_chain_instance

    service = RAGService(db_path="/fake/path")

    # CALL WITHOUT await — because real chat() is sync
    result = service.chat("Test query")

    assert isinstance(result, dict)
    assert result["answer"] == "Test answer"
    assert result["sources"] == ["doc1", "doc2"]