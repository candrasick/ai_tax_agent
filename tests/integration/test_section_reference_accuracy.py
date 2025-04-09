# tests/integration/test_section_reference_accuracy.py

import os
import sys
import pytest
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# Ensure the main project directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# LangChain & Project components
# Use try-except for imports that might fail if dependencies aren't perfect
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from ai_tax_agent.settings import settings
    from ai_tax_agent.database.models import IrsBulletinItem
    from ai_tax_agent.database.session import get_session
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain or project components not available, skipping tests: {e}")
    LANGCHAIN_AVAILABLE = False

# --- Test Configuration ---
NUM_ITEMS_TO_TEST = 5
MODEL_NAME = "gemini-1.5-flash-latest" # Or use gemini-1.5-pro for potentially higher accuracy

# --- Pydantic Schema for LLM Verification Output ---
class VerificationOutput(BaseModel):
    is_accurate: bool = Field(description="True if the provided section list accurately reflects the main sections discussed or referenced in the text, False otherwise.")
    reasoning: Optional[str] = Field(None, description="Brief explanation if inaccurate, or confirmation if accurate.")

# --- LLM Verification Prompt ---
VERIFICATION_SYSTEM_PROMPT = """You are an expert legal analyst. Your task is to verify the accuracy of extracted code section references from a given text segment (likely from an IRS bulletin item).

You will be given:
1. The original text segment.
2. A comma-delimited list of section numbers that were supposedly extracted from this text.

Please determine if the provided list accurately and reasonably represents the primary U.S. Code sections (especially Title 26) mentioned or discussed in the text segment. Minor omissions or inclusions of very peripheral references are acceptable, but the list should capture the main referenced sections.

Respond ONLY with the specified JSON format."""

VERIFICATION_HUMAN_TEMPLATE = """Text Segment:
```text
{full_text}
```

Extracted Section References:
`{referenced_sections}`

Is the list of Extracted Section References an accurate representation of the main sections mentioned in the Text Segment? Respond using the required JSON format, providing `is_accurate` (boolean) and optionally `reasoning` (string)."""

# --- Helper function to get test data (Not a fixture anymore) --- 
def _fetch_data_for_tests() -> List[Tuple[str, str, str]]:
    if not LANGCHAIN_AVAILABLE:
        # This check might be less effective here, better to keep on test function
        return [] 
        
    session: Session = get_session()
    items_data = []
    found_items = False
    try:
        items = session.query(IrsBulletinItem)\
                       .filter(IrsBulletinItem.full_text != None, 
                               IrsBulletinItem.referenced_sections != None)\
                       .order_by(IrsBulletinItem.id)\
                       .limit(NUM_ITEMS_TO_TEST)\
                       .all()
        
        for item in items:
            test_id = f"Item_{item.id}_{item.item_type}_{item.item_number}".replace(" ", "_").replace(".", "")
            # Return simple tuples now, pytest.param might interact poorly here
            items_data.append((item.full_text, item.referenced_sections or "", test_id))
            found_items = True
            
        if not found_items:
             # Cannot skip from here, test function should handle empty list
             print(f"Warning: Could not find {NUM_ITEMS_TO_TEST} suitable items in DB for testing.")
             
    except Exception as e:
        session.rollback()
        print(f"Warning: Database error during test data fetching: {e}")
        # Raise an error or return empty list depending on desired behavior
        # raise # Re-raise to fail collection
    finally:
        session.close()
    return items_data

# Fetch data *before* defining the tests that use it
TEST_DATA = _fetch_data_for_tests()

# --- The Test Function --- 
@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain components not available")
@pytest.mark.skipif(not TEST_DATA, reason="No suitable test data found in DB")
@pytest.mark.parametrize(("full_text", "ref_sections", "test_id"), TEST_DATA)
def test_reference_accuracy(full_text: str, ref_sections: str, test_id: str):
    """Uses Gemini to verify the accuracy of extracted section references."""
    print(f"\nTesting {test_id}...") # Print which item is being tested
    
    try:
        # Initialize LLM (consider making this a fixture if running many tests)
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            google_api_key=settings.gemini_api_key,
            temperature=0.1
        )
        
        # Setup Prompt and Parser
        parser = PydanticOutputParser(pydantic_object=VerificationOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", VERIFICATION_SYSTEM_PROMPT),
            ("human", VERIFICATION_HUMAN_TEMPLATE)
        ])
        
        # Create the verification chain
        chain = prompt | llm | parser
        
        # Invoke the chain
        verification_result: VerificationOutput = chain.invoke({
            "full_text": full_text,
            "referenced_sections": ref_sections
        })
        
        print(f"LLM Verification Result for {test_id}: Accurate={verification_result.is_accurate}, Reasoning='{verification_result.reasoning or ''}'")
        
        # Assertion: Check if the LLM verified the extraction as accurate
        assert verification_result.is_accurate, f"LLM indicated inaccurate extraction for {test_id}. Reasoning: {verification_result.reasoning}"
        
    except Exception as e:
        # Handle potential errors during LLM call or parsing
        pytest.fail(f"Error during LLM verification for {test_id}: {e}") 