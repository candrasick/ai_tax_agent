from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ai_tax_agent.settings import settings
import logging
from typing import List
from chromadb.api.types import EmbeddingFunction

logger = logging.getLogger(__name__)

# --- Embedding Function Wrapper --- #
class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """Wraps a LangChain embedding function to ensure ChromaDB compatibility for get_collection.
    ChromaDB validates the __call__ signature when an embedding function is passed to get_collection.
    """
    def __init__(self, langchain_embedder: GoogleGenerativeAIEmbeddings):
        self._langchain_embedder = langchain_embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embeds the input texts using the wrapped LangChain embedder's method."""
        return self._langchain_embedder.embed_documents(input)

def get_gemini_llm(model_name: str, temperature: float = 0.1) -> ChatGoogleGenerativeAI | None:
    """Initializes and returns a ChatGoogleGenerativeAI instance.

    Args:
        model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash-latest").
        temperature: The sampling temperature for the LLM.

    Returns:
        An initialized ChatGoogleGenerativeAI instance or None if initialization fails.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            convert_system_message_to_human=True # Often helpful for Gemini
        )
        logger.info(f"Initialized Google Gemini LLM with model: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM (model: {model_name}): {e}")
        logger.error("Please ensure your GEMINI_API_KEY is set correctly in the .env file and langchain-google-genai is installed.")
        return None

def get_embedding_function(model_name: str = "models/text-embedding-004", task_type: str = "retrieval_document") -> LangchainEmbeddingFunctionWrapper | None:
    """Initializes the Langchain embedder and returns it wrapped for ChromaDB get_collection compatibility.

    Args:
        model_name (str): The embedding model name.
        task_type (str): The task type for the embedding model.

    Returns:
        LangchainEmbeddingFunctionWrapper | None: A ChromaDB-compatible wrapped instance or None on error.
    """
    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable not set or not loaded into settings.")
        return None
    try:
        # Create the raw LangChain embedding function instance
        lc_embedding_function = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            task_type=task_type
        )
        logger.info(f"Initialized GoogleGenerativeAIEmbeddings with model: {model_name} and task type: {task_type}")

        # Return the WRAPPED instance
        return LangchainEmbeddingFunctionWrapper(lc_embedding_function)

    except Exception as e:
        logger.error(f"Failed to initialize or wrap GoogleGenerativeAIEmbeddings: {e}", exc_info=True)
        return None 