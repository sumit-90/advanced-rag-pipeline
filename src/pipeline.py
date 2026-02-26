from src.ingestion.document_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents
from src.indexing.vector_store import index_documents
from src.retrieval.retriever import retrieve_documents
from src.retrieval.reranker import rerank_documents
from src.generation.generator import generate_response
from src.guardrails.guardrails import validate_query, validate_response
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()

def run_ingestion_pipeline(directory:str) -> dict:
    try:
        logger.info(f"Starting ingestion pipeline for directory: {directory}")
        documents = load_documents(directory)
        logger.info(f"Loaded {len(documents)} documents from directory.")
        
        chunks = chunk_documents(documents)
        logger.info(f"Chunked documents into {len(chunks)} chunks.")

        embedded_chunks = embed_documents(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks.")

        indexing_result = index_documents(embedded_chunks)
        logger.info(f"Indexing result: {indexing_result}")

        result_summary = {
            "status":indexing_result['status'],
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "collection": indexing_result['collection']
        }
        
        return result_summary
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        return {"status": "failed", "documents_loaded": 0, "chunks_created": 0, "collection": ""}

def run_query_pipeline(query:str) -> dict:
    try:
        is_valid, reason = validate_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {reason}")
            return {"answer": reason, "sources": [], "model": ""}
        
        logger.info(f"Starting query pipeline for query: {query[:50]}...")
        retrieved_docs = retrieve_documents(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents for the query.")

        reranked_docs = rerank_documents(query, retrieved_docs)
        logger.info(f"Reranked documents, top score: {reranked_docs[0]['score'] if reranked_docs else 'N/A'}")

        response = generate_response(query, reranked_docs)
        logger.info(f"Generated response for the query.")
        
        result_summary = {
            "answer": response['answer'],
            "sources": response['sources'],
            "model": response['model']
        }
        is_valid, reason = validate_response(result_summary)
        if not is_valid:
            logger.warning(f"Response validation failed: {reason}")
        
        return result_summary
    except Exception as e:
        logger.error(f"Query pipeline failed: {e}")
        return {"answer": "Sorry, an error occurred while processing your query.", "sources": [], "model": ""}
