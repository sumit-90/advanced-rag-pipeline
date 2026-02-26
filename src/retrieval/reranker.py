import cohere
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()


def rerank_documents(query:str, documents:list[dict]) -> list[dict]:
    if not query:
        logger.warning("No query provided for reranking")
        return documents

    if not documents:
        logger.warning("No documents provided for reranking")
        return []

    cohere_client = cohere.Client(api_key=config['credentials']['cohere_api_key'])

    
    # Prepare the input for the reranking model
    inputs = [doc['content'] for doc in documents]


    # Get relevance scores from Cohere
    try:
        response = cohere_client.rerank(
            model=config['reranker']['model_name'],
            top_n=config['reranker']['top_n'],
            query=query,
            documents=inputs
        )
        reranked_docs = []
        for i, result in enumerate(response.results):
            doc = documents[result.index]
            reranked_docs.append({
                "content": doc['content'],
                "metadata": doc['metadata'],
                "original_score": doc['score'],
                "score": result.relevance_score
            })
        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Reranked to top {len(reranked_docs)} documents based on relevance scores.")
        return reranked_docs
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return documents