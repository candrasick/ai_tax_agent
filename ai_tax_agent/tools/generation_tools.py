# ai_tax_agent/tools/generation_tools.py
import logging
from langchain.tools import Tool
from pydantic import BaseModel, Field # For defining structured input
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Attempt to import get_gemini_llm, handle potential path issues
try:
    from ai_tax_agent.llm_utils import get_gemini_llm
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai_tax_agent.llm_utils import get_gemini_llm

logger = logging.getLogger(__name__)

# --- Simplify Tool ---

class SimplifyInput(BaseModel):
    """Input schema for the simplify_section_text tool."""
    section_text: str = Field(description="The original text of the tax code section to be simplified.")
    # Add other relevant context if needed, e.g., section_id, complexity_score
    # section_id: str = Field(description="The identifier of the section being simplified.")

def simplify_section_text(section_text: str, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1) -> str:
    """
    Analyzes the input section_text and generates a revised version focused
    on improving clarity and readability without changing the core meaning
    or structure significantly. Useful for minor rewording. Calls an LLM.

    Args:
        section_text: The original text of the tax code section.
        model_name: The Gemini model to use for simplification.
        temperature: The sampling temperature for the LLM.

    Returns:
        The simplified text string, or an error message if simplification fails.
    """
    logger.info(f"Attempting to simplify text (length: {len(section_text)} chars) using {model_name}...")

    # 1. Get LLM instance
    llm = get_gemini_llm(model_name=model_name, temperature=temperature)
    if not llm:
        error_msg = "Failed to initialize LLM. Cannot simplify text."
        logger.error(error_msg)
        return f"[Error: {error_msg}]"

    # 2. Define the simplification prompt as a direct string
    # Combining system instructions and user request into one input
    prompt_string = f"""System: You are an AI assistant specialized in simplifying complex legal and financial text, specifically sections of the US tax code. Your goal is to improve readability and clarity using plain language, while strictly preserving the original legal meaning, scope, and intent. Do NOT change the structure, add new information, remove essential details, or alter numerical values or specific legal terms unless absolutely necessary for simplification. Focus solely on rewording for better understanding by a non-expert audience. Output only the simplified text, without any preamble or explanation.

Human: Please simplify the following tax code section text:

{section_text}

Simplified Text:"""

    # 3. Invoke the LLM with the direct string prompt
    try:
        logger.debug("Invoking LLM for simplification...")
        response = llm.invoke(prompt_string) # Pass the formatted string directly
        simplified_text = response.content
        logger.info(f"Successfully simplified text. Output length: {len(simplified_text)} chars.")
        return simplified_text.strip() # Return the content, stripping whitespace
    except Exception as e:
        error_msg = f"Error during LLM invocation for simplification: {e}"
        logger.error(error_msg, exc_info=True)
        return f"[Error: {error_msg}]"

# Create the LangChain Tool
simplify_tool = Tool.from_function(
    func=simplify_section_text,
    name="Simplify Section Text",
    description="Generates a revised version of a tax section text, focusing on improving clarity and readability without significant structural changes. Use when minor rewording is needed. Input should be the original section text.",
    args_schema=SimplifyInput # Use the Pydantic model for input validation
)

# --- Redraft Tool (Placeholder for now) ---
# TODO: Implement Redraft tool later

# Example usage (for testing the tool function directly)
# Note: Requires GEMINI_API_KEY to be set in the environment/.env file
if __name__ == '__main__':
    # Configure logging for better output during testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check if API key is likely available (basic check)
    # In a real app, you'd use a settings management approach like Pydantic BaseSettings
    import os
    if not os.getenv("GEMINI_API_KEY"):
         print("\nWARNING: GEMINI_API_KEY environment variable not found.")
         print("LLM call in test mode will likely fail. Set the key in your .env file.")
         # Optionally exit or skip the LLM call test
         # exit() # Uncomment to stop if key is missing

    test_text = "Sec. 162. Trade or business expenses. (a) In general. There shall be allowed as a deduction all the ordinary and necessary expenses paid or incurred during the taxable year in carrying on any trade or business, including..."
    print("\n--- Testing simplify_section_text ---")
    result = simplify_section_text(section_text=test_text)
    print(f"\nInput Text:\n{test_text}\n")
    print(f"Output Simplified Text:\n{result}")
    print("\n--- Test Complete ---") 