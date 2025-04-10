from langchain_google_genai import ChatGoogleGenerativeAI
from ai_tax_agent.settings import settings
import logging

logger = logging.getLogger(__name__)

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