from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import uuid
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()


def index_documents(chunks:list[dict])->dict:
    if not chunks:
        logger.warning("No chunks provided for indexing")
        return {"status": "failed", "indexed_count": 0, "collection": config['vector_store']['collection_name']}

    client = QdrantClient(
        url=config['vector_store']['url'],
        prefer_grpc=True,
    )

    # Create collection if it doesn't exist
    if not client.has_collection(config['vector_store']['collection_name']):
        client.create_collection(
            collection_name=config['vector_store']['collection_name'],
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{config['vector_store']['collection_name']}' created.")

    else:
        logger.info(f"Collection '{config['vector_store']['collection_name']}' already exists.")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk['embedding'],
            payload={
                "content": chunk['content'],
                **chunk['metadata']
            }
        )
        for chunk in chunks
    ]
    client.upsert(collection_name=config['vector_store']['collection_name'], points=points)
    logger.info(f"Added {len(chunks)} documents to the vector store.")
    
    return {"status": "success", "indexed_count": len(chunks), "collection": config['vector_store']['collection_name']}