import logging
from src.config_loader import load_config
config = load_config()

def get_logger(name: str) -> logging.Logger:
    try:
        logger = logging.getLogger(name)
        
        # configure handler, format, level once here
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler(config["logging"]["file"])
        formatter = logging.Formatter(config["logging"]["format"])
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
            logger.addHandler(file_handler)

        logger.setLevel(config["logging"]["level"])
        
        return logger
    except Exception as e:
        raise Exception(f"Error setting up logger: {e}")