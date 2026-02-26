from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)

config = load_config()


def chunk_documents(documents:list[dict])->list[dict]:
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    strategy = config["chunking"]["strategy"]
    if strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"]
        )
    elif strategy == "character":
        text_splitter = CharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"]
        )
    else:
        logger.warning(f"Unknown chunking strategy: {strategy}, defaulting to character-based splitting")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"]
        )

    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.create_documents([doc['content']])
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'content': chunk.page_content,
                'metadata': {
                    "source": doc['metadata']['source'],
                    "file_type": doc['metadata']['file_type'],
                    "file_size_kb": doc['metadata']['file_size_kb'],
                    "num_pages": doc['metadata']['num_pages'],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    logger.info(f"Produced {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs