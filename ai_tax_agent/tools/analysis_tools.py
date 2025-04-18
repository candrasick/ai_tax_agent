# ai_tax_agent/tools/analysis_tools.py
import logging
import textstat
from langchain.tools import Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Estimate Complexity Tool ---

class ComplexityInput(BaseModel):
    """Input schema for the estimate_text_complexity tool."""
    text_to_analyze: str = Field(description="The text content (e.g., original or revised tax section) for which to estimate complexity.")

def estimate_text_complexity(text_to_analyze: str) -> float:
    """
    Estimates the complexity of the input text using standard readability metrics.
    Currently uses the Flesch Reading Ease score, where lower scores indicate
    higher complexity (more difficult to read). The score typically ranges
    from 0 (very complex) to 100 (very easy).

    Args:
        text_to_analyze: The text content to analyze.

    Returns:
        A float representing the estimated complexity score (Flesch Reading Ease).
        Returns a default high complexity score (e.g., 0.0) if analysis fails.
    """
    logger.info(f"Estimating complexity for text (length: {len(text_to_analyze)} chars)...")
    if not text_to_analyze or not isinstance(text_to_analyze, str) or len(text_to_analyze.split()) < 10:
         logger.warning("Text too short or invalid for meaningful complexity analysis. Returning default high complexity score.")
         # Return a score indicating very high complexity for short/invalid text
         # Flesch scale: lower = harder. 0 is very hard.
         return 0.0
    try:
        # Calculate Flesch Reading Ease score
        # Lower scores = more complex text
        flesch_ease_score = textstat.flesch_reading_ease(text_to_analyze)
        logger.debug(f"Calculated Flesch Reading Ease: {flesch_ease_score}")

        # The prompt expects a "complexity_score". While Flesch Ease measures *ease*,
        # using it directly is fine as long as the agent understands lower = more complex.
        # Alternatively, we could invert it (e.g., 100 - score), but let's keep it simple.
        complexity_score = float(flesch_ease_score)
        return complexity_score

    except Exception as e:
        logger.error(f"Error during text complexity analysis: {e}", exc_info=True)
        # Return a default score indicating high complexity on error
        return 0.0

# Create the LangChain Tool
estimate_complexity_tool = Tool.from_function(
    func=estimate_text_complexity,
    name="Estimate Text Complexity",
    description="Calculates a complexity score for a given text (like a tax section) using the Flesch Reading Ease metric. Lower scores mean higher complexity (harder to read). Input is the text to analyze.",
    args_schema=ComplexityInput
)

# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_complex_text = "Notwithstanding any other provision of this subchapter, the adjusted basis for determining the gain or loss from the sale or other disposition of property, whenever acquired, shall be the basis determined under section 1012 or other applicable sections of this subchapter, adjusted as provided in section 1016."
    test_simple_text = "The quick brown fox jumps over the lazy dog. This is a simple sentence. Readability should be high."
    short_text = "Too short."

    print("\n--- Testing estimate_text_complexity ---")

    score_complex = estimate_text_complexity(text_to_analyze=test_complex_text)
    print(f"\nComplex Text:\n'{test_complex_text}'")
    print(f"Complexity Score (Flesch Ease): {score_complex:.2f} (Lower is harder)")

    score_simple = estimate_text_complexity(text_to_analyze=test_simple_text)
    print(f"\nSimple Text:\n'{test_simple_text}'")
    print(f"Complexity Score (Flesch Ease): {score_simple:.2f} (Lower is harder)")

    score_short = estimate_text_complexity(text_to_analyze=short_text)
    print(f"\nShort Text:\n'{short_text}'")
    print(f"Complexity Score (Flesch Ease): {score_short:.2f} (Lower is harder)")

    print("\n--- Test Complete ---") 