# RAGentWeb/tests/test_rag_service.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.rag_service import RAGService

@patch("src.rag_service.FAISS.load_local", return_value=MagicMock())
@patch("src.rag_service.ConversationalRetrievalChain")
@pytest.mark.asyncio
async def test_rag_chat(mock_chain_class, mock_faiss):
    # Mock async chain
    mock_chain_instance = AsyncMock()
    mock_chain_instance.invoke = AsyncMock(return_value={
        "answer": "Test answer",
        "source_documents": ["doc1", "doc2"]
    })
    mock_chain_class.return_value = mock_chain_instance

    service = RAGService(db_path="/fake/path")
    result = await service.chat("Test query")
    
    assert isinstance(result, dict)
    assert result["answer"] == "Test answer"
    assert result["sources"] == ["doc1", "doc2"]
