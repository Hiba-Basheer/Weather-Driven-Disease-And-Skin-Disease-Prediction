# RAGentWeb/tests/test_rag_service.py
from unittest.mock import MagicMock, patch
from src.rag_service import RAGService

@patch("os.getenv", return_value="fake_key")
@patch("src.rag_service.FAISS.load_local", return_value=MagicMock())
@patch("src.rag_service.ConversationalRetrievalChain")
def test_rag_chat(mock_chain_class, mock_faiss, mock_getenv):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Dengue is a viral infection.",
        "source_documents": [
            MagicMock(metadata={"source": "doc1"}),
            MagicMock(metadata={"source": "doc2"})
        ]
    }
    mock_chain_class.from_llm.return_value = mock_chain

    service = RAGService(db_path="/fake")
    result = service.chat("What is dengue?")

    assert result["answer"] == "Dengue is a viral infection."
    assert result["sources"] == ["doc1", "doc2"]