# scripts/score_section_complexity.py

import argparse
import logging
import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Sequence

from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._common import GoogleGenerativeAIError

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings
from ai_tax_agent.database.session import get_session
# Import relevant models
from ai_tax_agent.database.models import UsCodeSection, SectionComplexity

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PROMPT_FILE_PATH = "prompts/score_section_complexity.txt"
DEFAULT_BATCH_SIZE = 50 # Commit changes every N records

# --- Helper Functions ---

def load_prompts_from_file(file_path: str) -> tuple[str, str]:
    """Loads system and human prompts from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        system_prompt_start = content.find("SYSTEM PROMPT:")
        human_prompt_start = content.find("HUMAN PROMPT:")

        if system_prompt_start == -1 or human_prompt_start == -1:
            raise ValueError("Could not find 'SYSTEM PROMPT:' or 'HUMAN PROMPT:' markers in the file.")

        system_prompt = content[system_prompt_start + len("SYSTEM PROMPT:"):human_prompt_start].strip()
        human_prompt = content[human_prompt_start + len("HUMAN PROMPT:"):].strip()

        if not system_prompt or not human_prompt:
             raise ValueError("System or Human prompt content is empty.")

        logger.info(f"Successfully loaded prompts from {file_path}")
        return system_prompt, human_prompt
    except FileNotFoundError:
        logger.error(f"Prompt file not found at: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading or parsing prompt file {file_path}: {e}")
        raise

def get_llm(settings):
    """Initializes the LLM."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    logger.debug("Initializing ChatGoogleGenerativeAI (gemini-1.5-flash)")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.gemini_api_key, temperature=0.1)

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Score US Code section complexity using an LLM and store results.")
    parser.add_argument("--clear", action="store_true", help="Clear the existing section_complexity table before scoring.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Commit results to DB every N sections.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)
    logger.info("--- Starting Section Complexity Scoring Agent ---")

    try:
        db: Session = get_session()
        logger.info("Database session acquired.")

        # Handle --clear flag
        if args.clear:
            logger.warning("Flag --clear specified. Deleting all records from section_complexity table...")
            try:
                deleted_count = db.execute(delete(SectionComplexity)).rowcount
                db.commit()
                logger.info(f"Deleted {deleted_count} records from section_complexity.")
            except Exception as e:
                logger.error(f"Error deleting records from section_complexity: {e}", exc_info=True)
                db.rollback()
                db.close()
                return # Stop if clearing fails

        # Load Prompts
        system_prompt, human_prompt = load_prompts_from_file(PROMPT_FILE_PATH)

        # Setup LLM and Chain
        llm = get_llm(settings)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        # Using StrOutputParser, we'll parse JSON manually
        chain = prompt_template | llm | StrOutputParser()
        logger.info("LLM and LangChain chain initialized.")

        # --- Resumability Logic ---
        logger.info("Fetching section IDs to process...")
        all_section_ids_query = select(UsCodeSection.id).order_by(UsCodeSection.id)
        all_section_ids_result = db.execute(all_section_ids_query).scalars().all()
        total_sections = len(all_section_ids_result)
        logger.info(f"Found {total_sections} total sections in us_code_section table.")

        existing_complexity_ids_query = select(SectionComplexity.section_id)
        existing_complexity_ids_result = db.execute(existing_complexity_ids_query).scalars().all()
        existing_ids_set = set(existing_complexity_ids_result)
        logger.info(f"Found {len(existing_ids_set)} sections already scored in section_complexity table.")

        ids_to_process = [id for id in all_section_ids_result if id not in existing_ids_set]
        logger.info(f"Need to process {len(ids_to_process)} sections.")

        if not ids_to_process:
            logger.info("All sections have already been scored. Exiting.")
            db.close()
            return
        # --- End Resumability Logic ---

        # Fetch sections that need processing in batches to manage memory (optional, adjust fetch_batch_size if needed)
        fetch_batch_size = 500 # How many sections to fetch from DB at once
        processed_count = 0
        error_count = 0
        start_time = time.time()

        for i in range(0, len(ids_to_process), fetch_batch_size):
            batch_ids = ids_to_process[i:i + fetch_batch_size]
            logger.info(f"Fetching details for section IDs {batch_ids[0]} to {batch_ids[-1]}...")

            sections_to_score: Sequence[UsCodeSection] = db.query(UsCodeSection).filter(
                UsCodeSection.id.in_(batch_ids)
            ).order_by(UsCodeSection.id).all() # Fetch full objects for this batch

            logger.info(f"Processing batch of {len(sections_to_score)} sections...")
            commit_counter = 0
            for section in tqdm(sections_to_score, desc=f"Scoring Batch {i//fetch_batch_size + 1}"):
                try:
                    # Prepare input, handle None values gracefully
                    input_data = {
                        "section_id": section.id,
                        "section_title": section.section_title or "N/A",
                        "core_text": section.core_text or "",
                        # Format floats for prompt clarity, handle None
                        "section_count_z": f"{section.section_count_z:.2f}" if section.section_count_z is not None else "N/A",
                        "amendment_count_z": f"{section.amendment_count_z:.2f}" if section.amendment_count_z is not None else "N/A",
                        "bulletins_count_z": f"{section.bulletins_count_z:.2f}" if section.bulletins_count_z is not None else "N/A",
                    }

                    # Invoke LLM Chain
                    # Add retries for LLM calls if needed
                    llm_response_str = chain.invoke(input_data)
                    logger.debug(f"LLM response for section {section.id}: {llm_response_str}")

                    # Parse and Validate JSON
                    try:
                        # Clean potential markdown fences
                        response_cleaned = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
                        result_json = json.loads(response_cleaned)

                        # Validate structure
                        if not isinstance(result_json, dict) or \
                           'section_id' not in result_json or \
                           'complexity_score' not in result_json or \
                           'rationale' not in result_json:
                            raise ValueError("Missing required keys in JSON response.")

                        # Validate types/values
                        if not isinstance(result_json['section_id'], int) or result_json['section_id'] != section.id:
                             logger.warning(f"Section ID mismatch in response for actual ID {section.id}. Got: {result_json.get('section_id')}. Using actual ID.")
                             # Continue, but we'll use the correct section.id

                        score = float(result_json['complexity_score'])
                        if not (0.0 <= score <= 10.0):
                            raise ValueError(f"Complexity score {score} out of range (0.0-10.0).")

                        rationale = str(result_json['rationale'])
                        if not rationale:
                             raise ValueError("Rationale cannot be empty.")

                        # Create DB Object
                        new_complexity = SectionComplexity(
                            section_id=section.id, # Use the correct ID from the loop
                            complexity_score=score,
                            rationale=rationale
                        )
                        db.add(new_complexity)
                        processed_count += 1
                        commit_counter += 1

                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response for section {section.id}: {llm_response_str}", exc_info=True)
                        error_count += 1
                    except (ValueError, TypeError) as e:
                        logger.error(f"Validation failed for section {section.id} response: {e}. Response: {llm_response_str}", exc_info=True)
                        error_count += 1

                except GoogleGenerativeAIError as e:
                    logger.error(f"Google API error processing section {section.id}: {e}", exc_info=True)
                    error_count += 1
                    # Consider adding retry logic here or stopping
                except Exception as e:
                    logger.error(f"Unexpected error processing section {section.id}: {e}", exc_info=True)
                    error_count += 1
                    # Consider stopping or continuing based on error severity

                # Commit batch to DB
                if commit_counter >= args.batch_size:
                    try:
                        logger.info(f"Committing batch of {commit_counter} results...")
                        db.commit()
                        logger.info("Batch committed.")
                        commit_counter = 0
                    except Exception as e:
                         logger.error(f"Database commit failed: {e}", exc_info=True)
                         db.rollback()
                         # Decide whether to stop or continue after commit failure

            # Commit any remaining items in the current fetch batch
            if commit_counter > 0:
                 try:
                    logger.info(f"Committing final {commit_counter} results for fetch batch...")
                    db.commit()
                    logger.info("Final batch committed.")
                 except Exception as e:
                    logger.error(f"Database commit failed for final batch: {e}", exc_info=True)
                    db.rollback()

        # --- Final Summary ---
        end_time = time.time()
        duration = end_time - start_time
        logger.info("--- Section Complexity Scoring Complete ---")
        logger.info(f"Total sections processed in this run: {processed_count}")
        logger.info(f"Total sections skipped due to errors: {error_count}")
        logger.info(f"Total duration: {duration:.2f} seconds")

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        if 'db' in locals() and db.is_active:
            db.rollback()
    finally:
        if 'db' in locals() and db:
            db.close()
            logger.info("Database session closed.")


if __name__ == "__main__":
    main() 