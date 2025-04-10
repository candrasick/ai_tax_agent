#!/usr/bin/env python
"""
Script to extract form field details from IRS form instructions using an LLM.
Navigates to the HTML instruction page, extracts main content, and uses an LLM
to identify and separate field labels (e.g., Line 1a.) and their full text.

Saves extracted fields to the form_fields table.
Includes options for limiting the number of forms processed and resuming.
"""

import os
import requests
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from typing import List, Optional

# --- LangChain / LLM Imports --- #
# from langchain_google_genai import ChatGoogleGenerativeAI # No longer needed directly
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Import from Pydantic v2
from langchain_core.output_parsers import PydanticOutputParser

# --- Shared Utilities --- #
from ai_tax_agent.llm_utils import get_gemini_llm # Import the shared function

# --- Database Imports --- #
from sqlalchemy.orm import Session, joinedload
from ai_tax_agent.database.models import FormInstruction, FormField, Base
from ai_tax_agent.database.session import get_session

# --- LLM Configuration --- #
# Define model and temperature for this script
MODEL_NAME = "gemini-1.5-flash-latest" # Switched from gemini-1.0-pro
LLM_TEMPERATURE = 0.0

# --- LLM Initialization --- #
# Ensure GOOGLE_API_KEY environment variable is set (handled by settings)
llm = get_gemini_llm(model_name=MODEL_NAME, temperature=LLM_TEMPERATURE)
# Note: Script will proceed but analyze_content_with_llm will fail if llm is None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('llm_form_field_extractor')

# --- Pydantic Schemas for LLM Output --- #

# Define Pydantic model for a single extracted field
class FormFieldOutput(BaseModel):
    field_label: str = Field(description="The exact label of the form field, e.g., 'Line 1a. Testate estates.'")
    full_text: str = Field(description="The complete text associated with this field label, including the label and following explanation.")

# Define Pydantic model for the list of fields extracted from one instruction
class InstructionAnalysisOutput(BaseModel):
    fields: List[FormFieldOutput] = Field(description="A list of all extracted fields and their text from the instruction document.")

# --- Output Parser --- #
try:
    parser = PydanticOutputParser(pydantic_object=InstructionAnalysisOutput)
    logger.info("PydanticOutputParser initialized.")
except Exception as e:
    logger.error(f"Failed to initialize PydanticOutputParser: {e}")
    parser = None

# --- Function Definitions --- #

def parse_arguments():
    """Parse command line arguments"""
    parser_cli = argparse.ArgumentParser(description="Extract form fields from IRS instructions using an LLM.")
    parser_cli.add_argument('--limit', type=int, default=None, 
                        help='Limit the number of form instructions to process (for testing).')
    # Add other arguments if needed (e.g., --model, --batch-size)
    return parser_cli.parse_args()

def fetch_and_parse_html(url: str) -> Optional[BeautifulSoup]:
    """Fetches HTML from a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, timeout=60) # Increased timeout
        response.raise_for_status() # Check for HTTP errors
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing HTML from {url}: {e}")
        return None

def extract_main_content(soup: BeautifulSoup) -> Optional[str]:
    """Extracts text content from the main content area of the IRS instructions page."""
    main_content = soup.select_one('#main-content')
    if not main_content:
        logger.warning("Could not find <div id='main-content'> in the HTML.")
        # Fallback: try getting all text from body if main-content fails
        body = soup.find('body')
        return body.get_text(separator='\n', strip=True) if body else None
    
    # Attempt to remove known navigation/irrelevant sections within main-content if necessary
    # Example: Find and remove left nav if it's inside #main-content
    # left_nav = main_content.select_one('.left-nav-class') # Replace with actual selector if needed
    # if left_nav: 
    #    left_nav.decompose()

    return main_content.get_text(separator='\n', strip=True)

def get_instructions_to_process(session: Session, limit: Optional[int] = None) -> List[FormInstruction]:
    """Gets FormInstruction records that haven't been processed yet."""
    query = session.query(FormInstruction)\
        .options(joinedload(FormInstruction.fields))\
        .filter(FormInstruction.html_url != None)\
        .order_by(FormInstruction.id)
                   
    all_instructions = query.all()
    
    instructions_to_process = []
    for instruction in all_instructions:
        if not instruction.fields: # Check if the fields relationship is empty
            instructions_to_process.append(instruction)
            
    logger.info(f"Found {len(instructions_to_process)} instructions without extracted fields out of {len(all_instructions)} total.")

    if limit is not None:
        logger.info(f"Applying limit: processing max {limit} instructions.")
        return instructions_to_process[:limit]
    else:
        return instructions_to_process

def analyze_content_with_llm(content: str) -> Optional[List[dict]]:
    """Uses the Gemini LLM chain to analyze content and extract form fields."""
    if not llm or not parser:
        # Error message now includes check for initialization failure
        logger.error("LLM or Parser not properly initialized. Cannot analyze content.")
        return None
        
    # --- LangChain prompt and chain execution --- # 
    logger.debug("Starting LLM analysis...")
    prompt_text = """Analyze the following IRS form instruction text. Identify each distinct line item or field explanation (e.g., starting with 'Line X.', 'Part Y.', 'Section Z:', etc.). 
For each identified item, extract two pieces of information:
1.  'field_label': The exact introductory label for the field, including the line number and title (e.g., "Line 1a. Testate estates.").
2.  'full_text': The complete text associated with that field label, including the label itself and all explanatory paragraphs following it, up until the beginning of the next field label or the end of the relevant section.

{format_instructions}

Instruction Text:
--- BEGIN TEXT ---
{instruction_text}
--- END TEXT ---

Extracted Fields:"""
    
    try:
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Ensure Gemini gets appropriate roles if needed, or use `convert_system_message_to_human=True` during init
        chain = prompt | llm | parser
        
        # Slice content if too long? Gemini might have token limits.
        # Consider chunking or summarizing if necessary for very long instructions.
        # max_tokens = 8000 # Example limit, check Gemini model's actual limit
        # if len(content) > max_tokens * 4: # Heuristic: avg 4 chars/token
        #    logger.warning(f"Content length ({len(content)}) might exceed token limit. Truncating.")
        #    content = content[:max_tokens * 4] 
            
        result: InstructionAnalysisOutput = chain.invoke({
            "instruction_text": content,
            "format_instructions": parser.get_format_instructions()
        })
        logger.debug(f"LLM analysis complete. Found {len(result.fields)} potential fields.")
        # Convert Pydantic models back to dictionaries for saving
        return [field.dict() for field in result.fields]
    except Exception as e:
        logger.error(f"Error during LLM analysis chain execution: {e}", exc_info=True)
        return None
    # --- End Chain Execution ---

def process_instruction(instruction: FormInstruction, session: Session):
    """Fetches, analyzes, and saves fields for a single instruction."""
    logger.info(f"Processing: {instruction.title} (ID: {instruction.id}, Form: {instruction.form_number}) - {instruction.html_url}")
    
    soup = fetch_and_parse_html(instruction.html_url)
    if not soup:
        return # Skip if fetching/parsing failed
        
    main_content = extract_main_content(soup)
    if not main_content or len(main_content) < 100: # Basic check for meaningful content
        logger.warning(f"Could not extract sufficient main content from {instruction.html_url}. Skipping.")
        return
        
    # Analyze content with LLM
    extracted_fields = analyze_content_with_llm(main_content)
    
    if extracted_fields is None: # Indicates an error during analysis
        logger.error(f"LLM analysis failed for {instruction.form_number}. Skipping saving.")
        return
        
    if not extracted_fields:
        logger.warning(f"LLM did not extract any fields for {instruction.form_number}.")
        return
        
    # Save extracted fields to database
    saved_count = 0
    try:
        for field_data in extracted_fields:
            # Add validation here if needed (e.g., check label format)
            if not field_data.get('field_label') or not field_data.get('full_text'):
                logger.warning(f"LLM returned incomplete field data: {field_data}. Skipping field.")
                continue

            new_field = FormField(
                instruction_id=instruction.id,
                field_label=field_data['field_label'],
                full_text=field_data['full_text']
            )
            session.add(new_field)
            saved_count += 1
            
        session.commit() # Commit fields for this instruction
        logger.info(f"Successfully saved {saved_count} fields for form {instruction.form_number}.")
        
    except Exception as e:
        logger.error(f"Database error saving fields for form {instruction.form_number}: {e}")
        session.rollback()

# --- Main Execution --- #

def main():
    """Main function to run the LLM field extractor."""
    logger.info("Starting LLM-based IRS form field extractor...")
    args = parse_arguments()
    
    session = get_session()
    
    try:
        instructions_to_process = get_instructions_to_process(session, args.limit)
        
        if not instructions_to_process:
            logger.info("No instructions found needing processing.")
            return
            
        logger.info(f"Starting processing for {len(instructions_to_process)} instructions.")
        
        for instruction in tqdm(instructions_to_process, desc="Analyzing Instructions"):
            process_instruction(instruction, session)
            
        logger.info("Processing complete.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during main processing: {e}", exc_info=True)
    finally:
        session.close()
        logger.info("Database session closed.")

if __name__ == "__main__":
    
    main() 