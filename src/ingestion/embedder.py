from langchain_openai.embeddings import OpenAIEmbeddings
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()

def embed_documents(documents:list[dict])->list[dict]:
    """Embed documents using OpenAI embeddings."""

    if not documents:
        logger.warning("No documents provided for embedding")
        return []
    else:
        api_key = config['credentials']['openai_api_key']
        embedding_model = OpenAIEmbeddings(
            model=config['embedding']['model_name'],
            openai_api_key=api_key
        )
        
        texts = [doc['content'] for doc in documents]
        embeddings = embedding_model.embed_documents(texts)   # one API call

        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding

        logger.info(f"total embedding generated for {len(documents)} documents")
        return documents