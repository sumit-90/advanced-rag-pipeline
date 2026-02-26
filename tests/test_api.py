from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ingest_success():
    mock_result = {"status": "success", "documents_loaded": 3, "chunks_created": 10, "collection": "rag_collection"}
    with patch("api.main.run_ingestion_pipeline", return_value=mock_result):
        response = client.post("/ingest", json={"directory": "test_docs"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_ingest_failure():
    mock_result = {"status": "failed", "documents_loaded": 0, "chunks_created": 0, "collection": ""}
    with patch("api.main.run_ingestion_pipeline", return_value=mock_result):
        response = client.post("/ingest", json={"directory": "test_docs"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Ingestion pipeline failed"


def test_query_success():
    mock_result = {"answer": "Attention is a mechanism in neural networks.", "sources": ["doc1.pdf"], "model": "gpt-4.1-mini"}
    with patch("api.main.run_query_pipeline", return_value=mock_result):
        response = client.post("/query", json={"query": "What is attention?"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("Attention is a mechanism")


def test_query_failure():
    mock_result = {"answer": "", "sources": [], "model": ""}
    with patch("api.main.run_query_pipeline", return_value=mock_result):
        response = client.post("/query", json={"query": "What is attention?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Query pipeline failed"
