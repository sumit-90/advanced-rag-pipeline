import os
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyMuPDFLoader
from src.config_loader import load_config
from src.logger import get_logger

config = load_config()
logger = get_logger(__name__)


def load_documents(directory:str)->list[dict]:
    dir_path = os.path.join(config["source_data"]["document_directory"], directory)
    if not os.listdir(dir_path):
        logger.warning(f"Directory is empty: {dir_path}")
        return []
    
    files = os.listdir(dir_path)
    file_paths = []
    for file in files:
        if file.endswith(('.txt', '.md', '.pdf')):
            file_paths.append(os.path.join(dir_path, file))
        else:
            logger.warning(f"Skipping unsupported file: {file}") 

    all_docs = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            docs = loader.load()
        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        else:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

    
        for doc in docs:
            all_docs.append({
                'content': doc.page_content,
                'metadata': {
                    "source": doc.metadata.get('source', 'unknown'),
                    "file_type": file_path.split('.')[-1],
                    "file_size_kb": os.path.getsize(file_path) / 1024,
                    "num_pages": len(docs) if file_path.endswith('.pdf') else 1
                }
            })
    file_cnt = len(file_paths)
    logger.info(f"Loaded {len(all_docs)} documents from {file_cnt} files in directory: {dir_path}")
    return all_docs