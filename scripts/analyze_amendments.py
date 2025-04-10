#!/usr/bin/env python
"""
Agent script to analyze U.S. Code sections, count amendments using an LLM,
separate amendment text from core text, and update the database.
"""

import os
import sys
import argparse
from datetime import date
from typing import Optional

from tqdm import tqdm
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# LangChain components
# from langchain_google_genai import ChatGoogleGenerativeAI # No longer needed directly
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Project components
from ai_tax_agent.settings import settings # Keep for now, maybe remove later if key only used in util
from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session
from ai_tax_agent.llm_utils import get_gemini_llm # Import the shared function

# --- LLM Configuration ---
MODEL_NAME = "gemini-1.5-flash-latest"
LLM_TEMPERATURE = 0.1 # Define temperature separately

# --- Pydantic Output Schema (Simplified & Field Renamed) ---
class AmendmentAnalysisOutput(BaseModel):
    """Schema for the LLM output, now only containing the estimated count."""
    amendment_count: int = Field(description="The estimated number of distinct amendments mentioned or detailed in the text.")
    # amendments_text and core_text removed, will be handled by heuristic

# --- Prompt Template (Simplified) ---
SYSTEM_PROMPT = """You are an expert legal assistant specialized in analyzing U.S. legislative text.
Your task is to analyze the provided full text of a U.S. Code section and estimate the number of distinct amendments it describes.

1.  Carefully read the entire text, paying attention to any parts that describe amendments (e.g., sections titled 'Amendments', 'Historical and Statutory Notes', specific notes detailing changes made by Public Laws).
2.  Estimate the total number of *distinct* amendments described within the text.
3.  Format your response strictly according to the provided JSON schema, providing a field named `amendment_count` with the estimated count.
"""

HUMAN_TEMPLATE = """Analyze the following U.S. Code section text and estimate the number of distinct amendments described:

```text
{section_text}
```

Please provide the estimated amendment count in the required JSON format using the key `amendment_count`.
"""

# --- Text Splitting Heuristic ---
HEURISTIC_SPLIT_HEADERS = [
    "Editorial Notes",
    "Amendments",
    "Historical and Statutory Notes",
    # Add other potential header variations if observed
]

def split_text_heuristically(full_text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Splits the full text into core and amendment sections based on heuristic headers.

    Args:
        full_text: The complete text of the section.

    Returns:
        A tuple containing (core_text, amendments_text).
        Returns (full_text, None) if no split point is found.
    """
    if not full_text:
        return None, None

    # Convert to lower case for case-insensitive matching
    lower_text = full_text.lower()
    split_index = -1

    # Find the first occurrence of any known header
    for header in HEURISTIC_SPLIT_HEADERS:
        try:
            index = lower_text.index(header.lower())
            if split_index == -1 or index < split_index:
                split_index = index
        except ValueError:
            continue # Header not found

    if split_index != -1:
        # Find the start of the line where the header was found
        # Go backwards from split_index until newline or start of string
        line_start_index = full_text.rfind('\n', 0, split_index)
        if line_start_index == -1: # Header might be on the first line
            line_start_index = 0
        else:
            line_start_index += 1 # Move past the newline

        core_text = full_text[:line_start_index].strip()
        amendments_text = full_text[line_start_index:].strip()
        return core_text, amendments_text
    else:
        # No split header found, assume all text is core text
        return full_text.strip(), None

# --- Database Functions (Update Function Modified) ---
def get_sections_to_analyze(session: Session, limit: Optional[int] = None) -> list[UsCodeSection]:
    """Fetches sections from the database that might need amendment analysis.
       For now, it fetches all sections, but could be refined later.
    """
    print("Fetching sections from database...")
    query = session.query(UsCodeSection).order_by(UsCodeSection.id)
    # Optional: Add filter criteria, e.g., where core_text is NULL
    # query = query.filter(UsCodeSection.core_text == None)
    if limit:
        query = query.limit(limit)
    sections = query.all()
    print(f"Found {len(sections)} sections to analyze.")
    return sections

def update_section_analysis(
    session: Session, 
    section_id: int, 
    amendment_count: int, 
    amendments_text: Optional[str], 
    core_text: str
):
    """Updates a specific section in the database with analysis results."""
    try:
        section = session.query(UsCodeSection).filter(UsCodeSection.id == section_id).first()
        if section:
            section.amendment_count = amendment_count
            section.amendments_text = amendments_text # Update new field
            section.core_text = core_text             # Update new field
            # section.full_text remains unchanged
            section.updated_at = date.today() # Update the timestamp
            # Don't commit here, commit in batches or at the end for performance
        else:
            print(f"Warning: Section ID {section_id} not found for updating.")
    except Exception as e:
        print(f"Error preparing update for section {section_id}: {e}")
        session.rollback() # Rollback potentially pending changes for safety

# --- Main Agent Logic (Updated) ---
def run_amendment_analysis(limit: Optional[int] = None, batch_size: int = 50):
    """Runs the amendment analysis agent using heuristic splitting and LLM counting."""
    print(f"Starting amendment analysis using model: {MODEL_NAME}")
    
    # Initialize LLM using the shared utility function
    llm = get_gemini_llm(model_name=MODEL_NAME, temperature=LLM_TEMPERATURE)
    if not llm:
        # Error is logged within get_gemini_llm
        return 

    # Setup Prompt and Parser
    parser = PydanticOutputParser(pydantic_object=AmendmentAnalysisOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE)
    ])
    
    # Create the analysis chain (LLM for counting only)
    chain = prompt | llm | parser
    
    # Get database session
    session = get_session()
    
    # Fetch sections
    sections = get_sections_to_analyze(session, limit)
    
    if not sections:
        print("No sections found to analyze.")
        session.close()
        return

    print(f"Analyzing {len(sections)} sections...")
    successful_updates = 0
    failed_updates = 0
    processed_count = 0
    
    # Process each section
    for section in tqdm(sections, desc="Analyzing Sections"):
        if not section.full_text or len(section.full_text.strip()) == 0:
            continue
            
        estimated_amendment_count_from_llm = 0 # Default count if LLM fails
        try:
            # 1. Invoke the LLM chain ONLY for counting
            analysis_result: AmendmentAnalysisOutput = chain.invoke({"section_text": section.full_text})
            # Access the count using the updated field name
            estimated_amendment_count_from_llm = analysis_result.amendment_count 
            
            # 2. Split text using heuristic
            core_text, amendments_text = split_text_heuristically(section.full_text)
            
            # 3. Prepare the database update
            update_section_analysis(
                session=session,
                section_id=section.id,
                # Use the count obtained from the LLM
                amendment_count=estimated_amendment_count_from_llm, 
                amendments_text=amendments_text,         # Text from heuristic
                core_text=core_text                     # Text from heuristic
            )
            successful_updates += 1
            
        except Exception as e:
            # Covers errors from LLM invoke/parse or heuristic split or DB update prep
            print(f"\nError processing section {section.id} (section_number: {section.section_number}): {e}")
            failed_updates += 1
            session.rollback() # Rollback potential changes for this section
        
        processed_count += 1
        # Commit periodically
        if processed_count % batch_size == 0:
            try:
                print(f"Committing batch of {batch_size} updates...")
                session.commit()
            except Exception as e:
                print(f"Error committing batch: {e}")
                session.rollback()

    # Commit any remaining changes
    try:
        print("Committing final batch...")
        session.commit()
    except Exception as e:
        print(f"Error committing final batch: {e}")
        session.rollback()
            
    print("\nAnalysis complete.")
    print(f"Successfully processed for update: {successful_updates}")
    print(f"Failed processing: {failed_updates}")

    # Close the session
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze U.S. Code sections for amendments using LLM for counting and heuristics for splitting.")
    parser.add_argument("-l", "--limit", type=int, default=None, 
                        help="Limit the number of sections to process (for testing). Default: process all.")
    parser.add_argument("--batch-size", type=int, default=50, 
                        help="Number of records to process before committing to the database.")
    # Removed --max-retries argument
    
    args = parser.parse_args()
    
    run_amendment_analysis(limit=args.limit, batch_size=args.batch_size) 