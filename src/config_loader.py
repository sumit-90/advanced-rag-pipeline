import os
import yaml
from dotenv import load_dotenv

def load_config(path: str = "./config/config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            config =  yaml.safe_load(f)
        
        load_dotenv()
        config["credentials"]["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        config["credentials"]["cohere_api_key"] = os.getenv("COHERE_API_KEY")

        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
