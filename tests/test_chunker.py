from src.ingestion.chunker import chunk_documents


def test_chunk_empty_input():
    result = chunk_documents([])
    assert result == []

def test_chunk_single_document():
    documents = [{"content": "This is a test document. It has multiple sentences.", "metadata": {"source": "test_doc.txt", "file_type": "txt", "file_size_kb": 1.0, "num_pages": 1}}]
    result = chunk_documents(documents)
    assert isinstance(result, list)
    assert len(result) > 0

def test_chunk_metadata_preserved():
    documents = [{"content": "This is a test document. It has multiple sentences.", "metadata": {"source": "test_doc.txt", "file_type": "txt", "file_size_kb": 1.0, "num_pages": 1}}]
    result = chunk_documents(documents)
    assert result[0]["metadata"]["source"] == "test_doc.txt"
    assert result[0]["metadata"]["file_type"] == "txt"

def test_chunk_index_added():
    documents = [{"content": "This is a test document. It has multiple sentences.", "metadata": {"source": "test_doc.txt", "file_type": "txt", "file_size_kb": 1.0, "num_pages": 1}}]
    result = chunk_documents(documents)
    assert "chunk_index" in result[0]["metadata"]
    assert "total_chunks" in result[0]["metadata"]
