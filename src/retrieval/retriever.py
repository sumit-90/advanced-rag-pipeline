from qdrant_client import QdrantClient
from langchain_openai.embeddings import OpenAIEmbeddings
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()


def retrieve_documents(query:str) -> list[dict]:
    if not query:
        logger.warning("No query provided for retrieval")
        return []

    client = QdrantClient(
        url=config['vector_store']['url'],
        prefer_grpc=True,
    )

    if not client.has_collection(config['vector_store']['collection_name']):
        logger.warning(f"Collection '{config['vector_store']['collection_name']}' does not exist.")
        return []

    # Generate embedding for the query
    embedding_model = OpenAIEmbeddings(model=config["embedding"]["model_name"], openai_api_key=config["credentials"]['openai_api_key'])
    query_embedding = embedding_model.embed_query(query)

    # Search for similar documents
    search_result = client.search(
        collection_name=config['vector_store']['collection_name'],
        query_vector=query_embedding,
        limit=config['retriever']['top_k']
    )
    

    retrieved_docs = []
    for hit in search_result:
        retrieved_docs.append({
            "content": hit.payload.get("content", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
            "score": hit.score,
        })
    results = [r for r in retrieved_docs if r['score'] >= config['retriever']['similarity_threshold']]

    logger.info(f"Retrieved {len(retrieved_docs)} documents for the query.")
    logger.info(f"Filtered to {len(results)} documents based on similarity threshold.")
    
    return results