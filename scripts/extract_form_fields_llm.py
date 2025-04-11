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
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
import logging
from typing import List, Optional

# --- LangChain / LLM Imports --- #
# from langchain_google_genai import ChatGoogleGenerativeAI # No longer needed directly
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Import from Pydantic v2
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException # Import the exception

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

# Reintroduce wrapper model, now expecting List[FormFieldOutput]
class InstructionAnalysisOutput(BaseModel):
    fields: List[FormFieldOutput] = Field(description="A list of all extracted fields and their text from the instruction document.")

# --- Output Parser --- #
try:
    # Parser now expects the wrapper model again
    parser = PydanticOutputParser(pydantic_object=InstructionAnalysisOutput)
    logger.info("PydanticOutputParser initialized to expect InstructionAnalysisOutput.")
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

def analyze_content_with_llm(content: str, instruction_form_number: str = "Unknown") -> Optional[List[dict]]:
    """Uses the Gemini LLM chain to analyze content and extract form fields.
       Validates each extracted field JSON string individually.
    """
    if not llm or not parser:
        # Error message now includes check for initialization failure
        logger.error("LLM or Parser not properly initialized. Cannot analyze content.")
        return None
        
    # --- Log Input Content --- 
    logger.debug(f"--- Analyzing Content for Form {instruction_form_number} (length: {len(content)}) ---")
    # Log first/last N chars for brevity, or full content if debugging needed
    # logger.debug(content) # Uncomment for full content logging if needed
    logger.debug(f"CONTENT START: {content[:500]}...") 
    logger.debug(f"...CONTENT END: ...{content[-500:]}") 
    # -------------------------
        
    # --- LangChain prompt and chain execution --- # 
    logger.debug("Starting LLM analysis...")
    # PROMPT: Ask for JSON object containing a list of objects
    prompt_text = """You are an expert assistant analyzing IRS form instruction text.
Your goal is to identify and extract distinct instructional items, typically corresponding to lines or specific sections on the associated tax form.

Look for headings like 'Line X', 'Line Xa', 'Lines X-Y', 'Part Z', or specific named sections (e.g., 'Who Must File', 'Specific Instructions').

For each distinct item identified, create a JSON object with two keys:
1.  'field_label': The exact introductory label or heading for the item (e.g., "Line 1a. Testate estates.", "Who Must File", "Line 16"). Capture the complete label text as presented.
2.  'full_text': The complete text associated *only* with that specific item, including the label/heading itself and all explanatory paragraphs belonging to it, stopping before the next distinct item's label/heading begins.

Output ONLY a single JSON object with one key, "fields", which contains a standard JSON list of the JSON objects you created for each item.

Example:
`{{
  "fields": [
    {{"field_label": "Line 1a. Testate estates.", "full_text": "Line 1a. Testate estates. Check the box..."}},
    {{"field_label": "Line 1b. Intestate estates.", "full_text": "Line 1b. Intestate estates. Check the box..."}}
  ]
}}`

{format_instructions}

Instruction Text to Analyze:
--- BEGIN TEXT ---
{instruction_text}
--- END TEXT ---

JSON object with extracted fields:"""
    
    try:
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | parser # Parser now expects InstructionAnalysisOutput
        
        # Slice content if too long? Gemini might have token limits.
        # Consider chunking or summarizing if necessary for very long instructions.
        # max_tokens = 8000 # Example limit, check Gemini model's actual limit
        # if len(content) > max_tokens * 4: # Heuristic: avg 4 chars/token
        #    logger.warning(f"Content length ({len(content)}) might exceed token limit. Truncating.")
        #    content = content[:max_tokens * 4] 
            
        # LLM should return an object parseable into InstructionAnalysisOutput
        result: InstructionAnalysisOutput = chain.invoke({
            "instruction_text": content,
            "format_instructions": parser.get_format_instructions()
        })
        logger.debug(f"LLM analysis and parsing complete. Found {len(result.fields)} fields.")

        # Convert the validated Pydantic objects (result.fields) to dictionaries
        return [field.model_dump() for field in result.fields]

    except OutputParserException as ope: # Catch if parsing InstructionAnalysisOutput fails
        logger.error(f"Failed to parse LLM output into InstructionAnalysisOutput: {ope}")
        return None
    except Exception as e:
        logger.error(f"Error during LLM analysis chain execution or validation: {e}", exc_info=True)
        return None
    # --- End Chain Execution ---

def process_instruction(instruction: FormInstruction, session: Session):
    """Fetches, chunks by HTML headings, analyzes, and saves fields."""
    logger.info(f"Processing: {instruction.title} (ID: {instruction.id}, Form: {instruction.form_number}) - {instruction.html_url}")
    
    soup = fetch_and_parse_html(instruction.html_url)
    if not soup:
        return
        
    # Get the main content Tag object
    main_content_tag = extract_main_content(soup)
    if not main_content_tag:
        logger.warning(f"Could not extract main content tag from {instruction.html_url}. Skipping.")
        return
        
    # --- Chunking Step using HTML --- 
    content_chunks = chunk_by_html_headings(main_content_tag, max_chars=20000) 
    if not content_chunks:
        logger.warning(f"HTML content chunking resulted in zero chunks for {instruction.html_url}. Skipping.")
        return
        
    all_valid_fields = [] # Aggregate results from all chunks
    analysis_failed = False
    
    # --- Process Chunks --- 
    logger.info(f"Processing content in {len(content_chunks)} chunk(s) for {instruction.form_number}.")
    for i, chunk in enumerate(content_chunks):
        logger.debug(f"Analyzing chunk {i+1}/{len(content_chunks)} (length: {len(chunk)} chars)")
        # Analyze content chunk with LLM - PASS FORM NUMBER
        validated_fields_from_chunk = analyze_content_with_llm(chunk, instruction.form_number)
        
        if validated_fields_from_chunk is None: # Indicates an error during analysis for this chunk
            logger.error(f"LLM analysis failed for chunk {i+1} of {instruction.form_number}. Skipping further analysis for this instruction.")
            analysis_failed = True
            break # Stop processing chunks for this instruction if one fails
            
        if validated_fields_from_chunk:
            all_valid_fields.extend(validated_fields_from_chunk)
            logger.debug(f"Added {len(validated_fields_from_chunk)} fields from chunk {i+1}.")
        else:
            logger.debug(f"Chunk {i+1} yielded no valid fields.")
    # --- End Chunk Processing ---
    
    # If analysis failed for any chunk, don't save anything for this instruction
    if analysis_failed:
        logger.error(f"Skipping save for {instruction.form_number} due to analysis failure in one or more chunks.")
        return
                
    if not all_valid_fields:
        logger.warning(f"LLM did not extract any valid/complete fields for {instruction.form_number} after processing all chunks.")
        return
        
    # Save ALL valid extracted fields to database
    saved_count = 0
    try:
        logger.info(f"Attempting to save {len(all_valid_fields)} aggregated fields for form {instruction.form_number}.")
        for field_data in all_valid_fields: # Iterate over the aggregated list
            new_field = FormField(
                instruction_id=instruction.id,
                field_label=field_data['field_label'],
                full_text=field_data['full_text']
            )
            session.add(new_field)
            saved_count += 1
            
        session.commit() # Commit all fields for this instruction together
        logger.info(f"Successfully saved {saved_count} validated fields for form {instruction.form_number}.")
        
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