# scripts/identify_exemptions.py

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
from ai_tax_agent.database.models import UsCodeSection, Exemption

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PROMPT_FILE_PATH = "prompts/identify_exemptions.txt"
DEFAULT_BATCH_SIZE = 20 # Commit changes every N sections processed (adjust based on # exemptions per section)

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
    # Using flash for potentially faster/cheaper analysis
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.gemini_api_key, temperature=0.0) # Low temp for factual extraction

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Identify potential exemptions in US Code sections using an LLM.")
    parser.add_argument("--clear", action="store_true", help="Clear the existing exemptions table before identifying.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Commit results to DB every N sections processed.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sections to process (for testing).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)
    logger.info("--- Starting Exemption Identification Agent ---")

    try:
        db: Session = get_session()
        logger.info("Database session acquired.")

        # Handle --clear flag
        if args.clear:
            logger.warning("Flag --clear specified. Deleting all records from exemptions table...")
            try:
                deleted_count = db.execute(delete(Exemption)).rowcount
                db.commit()
                logger.info(f"Deleted {deleted_count} records from exemptions.")
            except Exception as e:
                logger.error(f"Error deleting records from exemptions: {e}", exc_info=True)
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
        chain = prompt_template | llm | StrOutputParser()
        logger.info("LLM and LangChain chain initialized.")

        # --- Resumability Logic ---
        logger.info("Fetching section IDs to process...")
        # Select sections that have core_text
        all_section_ids_query = select(UsCodeSection.id).where(
            UsCodeSection.core_text.isnot(None),
            UsCodeSection.core_text != ""
        ).order_by(UsCodeSection.id)
        all_section_ids_result = db.execute(all_section_ids_query).scalars().all()
        total_sections = len(all_section_ids_result)
        logger.info(f"Found {total_sections} total sections with core text.")

        # Find sections already processed (have entries in exemptions table)
        existing_exemption_section_ids_query = select(Exemption.section_id).distinct()
        existing_exemption_section_ids_result = db.execute(existing_exemption_section_ids_query).scalars().all()
        existing_ids_set = set(existing_exemption_section_ids_result)
        logger.info(f"Found {len(existing_ids_set)} sections already have entries in the exemptions table.")

        ids_to_process = [id for id in all_section_ids_result if id not in existing_ids_set]

        # Apply limit if provided
        if args.limit is not None and args.limit < len(ids_to_process):
             logger.info(f"Applying limit: Processing only the first {args.limit} sections.")
             ids_to_process = ids_to_process[:args.limit]

        logger.info(f"Need to process {len(ids_to_process)} sections.")

        if not ids_to_process:
            logger.info("All relevant sections have already been processed. Exiting.")
            db.close()
            return
        # --- End Resumability Logic ---

        processed_count = 0
        error_count = 0
        exemptions_found_count = 0
        start_time = time.time()
        commit_counter = 0

        # Process sections one by one (LLM call per section)
        for section_id in tqdm(ids_to_process, desc="Identifying Exemptions"):
            try:
                # Fetch the specific section data needed
                section = db.get(UsCodeSection, section_id)
                if not section or not section.core_text:
                    logger.warning(f"Skipping section ID {section_id}: Not found or no core_text.")
                    continue

                # Prepare input
                input_data = {
                    "section_id": section.id,
                    "section_title": section.section_title or "N/A",
                    "core_text": section.core_text
                }

                # Invoke LLM Chain
                llm_response_str = chain.invoke(input_data)
                logger.debug(f"LLM response for section {section.id}: {llm_response_str}")

                # Parse and Validate JSON response (list of exemptions)
                try:
                    response_cleaned = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
                    found_exemptions = json.loads(response_cleaned)

                    if not isinstance(found_exemptions, list):
                        raise ValueError("LLM response is not a JSON list.")

                    # Process each found exemption object
                    if found_exemptions: # Only proceed if the list is not empty
                        for ex_data in found_exemptions:
                            if not isinstance(ex_data, dict) or \
                               'rationale' not in ex_data or \
                               'relevant_text' not in ex_data:
                                logger.warning(f"Invalid exemption object format for section {section.id}: {ex_data}. Skipping this exemption.")
                                continue

                            rationale = str(ex_data['rationale']).strip()
                            relevant_text = str(ex_data['relevant_text']).strip()

                            if not rationale or not relevant_text:
                                logger.warning(f"Empty rationale or relevant_text for section {section.id}: {ex_data}. Skipping this exemption.")
                                continue

                            # Create DB Object
                            new_exemption = Exemption(
                                section_id=section.id,
                                rationale=rationale,
                                relevant_text=relevant_text
                                # revenue_impact_estimate and entity_impact are left NULL
                            )
                            db.add(new_exemption)
                            exemptions_found_count += 1
                            logger.debug(f"  Added exemption for section {section.id}: {rationale[:50]}...")
                    else:
                        logger.debug(f"No exemptions identified by LLM for section {section.id}.")

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response for section {section.id}: {llm_response_str}", exc_info=True)
                    error_count += 1
                    continue # Skip to next section if parsing fails
                except (ValueError, TypeError) as e:
                    logger.error(f"Validation failed for section {section.id} response: {e}. Response: {llm_response_str}", exc_info=True)
                    error_count += 1
                    continue # Skip to next section if validation fails

                processed_count += 1
                commit_counter += 1

            except GoogleGenerativeAIError as e:
                logger.error(f"Google API error processing section {section.id}: {e}", exc_info=True)
                error_count += 1
                # Potentially break or sleep/retry here
            except Exception as e:
                logger.error(f"Unexpected error processing section {section.id}: {e}", exc_info=True)
                error_count += 1
                db.rollback() # Rollback potentially bad state for this section

            # Commit batch to DB
            if commit_counter >= args.batch_size:
                try:
                    logger.info(f"Committing batch of {commit_counter} processed sections ({exemptions_found_count} new exemptions recorded since last commit)...")
                    db.commit()
                    logger.info("Batch committed.")
                    commit_counter = 0
                    # Reset exemptions_found_count after commit to reflect batch count accurately?
                    # Let's keep it accumulating for the whole run for now.
                except Exception as e:
                     logger.error(f"Database commit failed: {e}", exc_info=True)
                     db.rollback()
                     # Decide whether to stop or continue after commit failure

        # Commit any remaining items
        if commit_counter > 0:
             try:
                logger.info(f"Committing final {commit_counter} processed sections...")
                db.commit()
                logger.info("Final commit successful.")
             except Exception as e:
                logger.error(f"Database commit failed for final items: {e}", exc_info=True)
                db.rollback()

        # --- Final Summary ---
        end_time = time.time()
        duration = end_time - start_time
        logger.info("--- Exemption Identification Complete ---")
        logger.info(f"Total sections processed in this run: {processed_count}")
        logger.info(f"Total new exemptions saved: {exemptions_found_count}")
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