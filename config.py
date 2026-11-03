# Standart library imports
import os


class LLMConfig:
    """
    Configuration class for LLM (Language Learning Model).
    This class holds all the parameters for the model configuration.
    """

    TEMPERATURE = 0.0

    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    SYSTEM_PROMPT = """If user asks to get a current date execute get_current_date tool.
Otherwise, return a text summary of user input - short representation of the text.

# ** Summary format to use **:
[summary]: ..."""


class APIConfig:
    """
    Configuration class for the API.
    This class holds the parameters for the API configuration.
    """

    HOST = os.getenv("API_HOST", "http://localhost:8000")


class SemanticSearchConfig:
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    PRODUCTS_CSV_PATH = "./data/products.csv"
    FAISS_INDEX_PATH = "./data/products.index"

    # The *-instruct E5 models expect queries as "Instruct: <task>\nQuery: <query>"; documents get no prefix
    QUERY_PREFIX = "Instruct: Given a product search query, retrieve relevant products\nQuery: "
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.8
