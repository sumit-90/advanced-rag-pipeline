from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()


def validate_query(query:str) -> tuple[bool, str]:
    logger.info(f"Validating query: {query[:50]}...")
    if not query:
        return False, "Query cannot be empty."
    
    if len(query) < config['validation']['min_query_length']:
        return False, f"Query must be at more than {config['validation']['min_query_length']} characters long."
    
    if len(query) > config['validation']['max_query_length']:
        return False, f"Query exceeds maximum length of {config['validation']['max_query_length']} characters."
    
    if query.strip().isnumeric():
        return False, "Query must contain more than just alphanumeric characters."
    
    # Add more validation rules as needed
    
    return True, "Query is valid."


def validate_response(response: dict) -> tuple[bool, str]:
    logger.info("Validating response...")

    # Check 1 — empty answer
    if not response.get('answer'):
        return False, "Empty answer returned."

    # Check 2 — no sources
    if not response.get('sources'):
        return False, "No sources returned."

    # Check 3 — LLM said it doesn't know
    if "i don't know" in response.get('answer', '').lower():
        return False, "LLM could not find answer in context."

    return True, "Response is valid."
