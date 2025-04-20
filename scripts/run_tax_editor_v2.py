# scripts/run_tax_editor.py
import logging
import argparse
import json
import os
import sys
from decimal import Decimal
from typing import Optional
import warnings
import re # Add import for regex

# Setup project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress the specific deprecation warning from langchain_google_genai
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Convert_system_message_to_human will be deprecated!.*",
    module="langchain_google_genai.*" # Target the specific module
)

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc # For ordering

# Progress bar import
from tqdm import tqdm

# Database imports
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import (
    UsCodeSection, 
    UsCodeSectionRevised, 
    SectionImpact, 
    SectionHistory,
    Exemption,  # Add import for Exemption model
)
from ai_tax_agent.database.versioning import determine_version_numbers

# Agent imports
from ai_tax_agent.agents import create_tax_editor_agent # Keep agent import
from ai_tax_agent.outputs import TaxEditorOutput # Import the output model from the new file
from ai_tax_agent.settings import settings # May need settings for agent config

# LangChain Parser import
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException # Import for specific error handling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def clear_working_version_edits(db: Session, working_version: int):
    """Deletes SectionVersion records for the current working version."""
    logger.info(f"Clearing existing edits for working version {working_version}...")
    try:
        deleted_count = db.query(UsCodeSectionRevised).filter(UsCodeSectionRevised.version == working_version).delete()
        db.commit()
        logger.info(f"Successfully deleted {deleted_count} UsCodeSectionRevised records for version {working_version}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear edits for version {working_version}: {e}", exc_info=True)
        raise


def process_sections(limit: int | None = None, clear: bool = False):
    """
    Iteratively processes tax sections using the tax editor agent.

    Args:
        limit: Maximum number of sections to process in this run.
        clear: Whether to clear existing progress for the current working version.
    """
    # Initial setup (versions, agent) - No DB session needed yet
    prior_version, working_version = determine_version_numbers()
    logger.info(f"Determined versions: Prior={prior_version}, Working={working_version}")

    if working_version == 0:
        logger.error("Working version is 0. Cannot run editor agent without an initial version.")
        return

    # --- Optional: Clear previous edits in its own session ---
    if clear:
        temp_db: Session | None = None
        try:
            temp_db = get_session()
            if not temp_db:
                logger.error("Failed to get database session for clearing edits.")
                return # Or handle error appropriately
            clear_working_version_edits(temp_db, working_version)
            # Re-determine versions might be needed if clearing changes things fundamentally
            # prior_version, working_version = determine_version_numbers()
        except Exception as clear_e:
            logger.error(f"Error during clearing of version {working_version}: {clear_e}", exc_info=True)
            # Decide if we should stop or continue if clearing fails
            return # Stop for now if clearing fails
        finally:
            if temp_db:
                temp_db.close()

    # --- Create Agent (outside loop) ---
    logger.info("Creating Tax Editor Agent...")
    agent_executor = create_tax_editor_agent(llm_model_name="gemini-1.5-pro", verbose=False)
    if not agent_executor:
        logger.error("Failed to create Tax Editor Agent.")
        return
    logger.info("Tax Editor Agent created successfully.")

    # --- Identify Sections to Process (using a temporary session) ---
    sections_to_process = []
    temp_db: Session | None = None
    try:
        temp_db = get_session()
        if not temp_db:
            logger.error("Failed to get database session for identifying sections.")
            return

        # First, get all sections that were deleted in any prior version
        deleted_sections = temp_db.query(UsCodeSectionRevised.orig_section_id)\
            .filter(UsCodeSectionRevised.deleted == True)\
            .filter(UsCodeSectionRevised.version < working_version)\
            .distinct()

        # Exclude both already processed sections in current version AND previously deleted sections
        subquery = temp_db.query(UsCodeSectionRevised.orig_section_id)\
            .filter(UsCodeSectionRevised.version == working_version)
        
        query = temp_db.query(UsCodeSection)\
            .filter(UsCodeSection.id.notin_(subquery))\
            .filter(UsCodeSection.id.notin_(deleted_sections))\
            .order_by(UsCodeSection.section_number)
            
        if limit is not None and limit > 0:
            query = query.limit(limit)
        sections_to_process = query.all()
    except Exception as query_e:
        logger.error(f"Failed to query sections to process: {query_e}", exc_info=True)
        return # Cannot proceed without sections
    finally:
        if temp_db:
            temp_db.close()

    if not sections_to_process:
        logger.info(f"No sections found needing processing for version {working_version}.")
        return

    logger.info(f"Found {len(sections_to_process)} sections to process (limit={limit}).")

    # --- Processing Loop (Each section gets its own session) ---
    processed_count = 0
    disable_tqdm = locals().get('args') and locals().get('args').quiet

    for section in tqdm(sections_to_process, desc="Processing Sections", unit="section", disable=disable_tqdm):
        db: Session | None = None # Session defined inside the loop
        try:
            # Create session for this section
            db = get_session()
            if not db:
                 logger.error(f"Failed to get database session for section {section.id}. Skipping.")
                 continue

            if not disable_tqdm:
                 logger.info(f"Processing Section ID: {section.id} ({section.section_number}) with session {db}")

            # 1. Get Input Text (from prior version)
            input_text: Optional[str] = None
            if prior_version == 0:
                original_section = db.query(UsCodeSection.core_text).filter(UsCodeSection.id == section.id).scalar()
                input_text = original_section
                logger.debug(f"Fetched original text (v0) for section {section.id}. Length: {len(input_text) if input_text else 0}")
            elif prior_version > 0:
                revised_section_text = db.query(UsCodeSectionRevised.core_text)\
                                         .filter(UsCodeSectionRevised.orig_section_id == section.id,
                                                 UsCodeSectionRevised.version == prior_version)\
                                         .scalar()
                input_text = revised_section_text
                logger.debug(f"Fetched revised text (v{prior_version}) for section {section.id}. Length: {len(input_text) if input_text else 0}")

            if input_text is None:
                 logger.warning(f"Could not find text for prior version {prior_version} for section {section.id}. Skipping.")
                 continue

            # 2. Get Additional Context
            impact = None
            revenue_impact_str = "N/A"
            
            if prior_version == 0:
                # For version 0, get original impact from SectionImpact table
                impact = db.query(SectionImpact).filter(SectionImpact.section_id == section.id).first()
                revenue_impact_str = f"{impact.revenue_impact / 1_000_000:.1f}" if impact and impact.revenue_impact is not None else "N/A"
            else:
                # For versions > 0, get revised impact from prior version
                prior_revision = db.query(UsCodeSectionRevised)\
                    .filter(UsCodeSectionRevised.orig_section_id == section.id)\
                    .filter(UsCodeSectionRevised.version == prior_version)\
                    .first()
                if prior_revision and prior_revision.revised_financial_impact is not None:
                    revenue_impact_str = f"{prior_revision.revised_financial_impact / 1_000_000:.1f}"
                    
            # Get current complexity score from the most recent version
            current_complexity_score = "N/A"
            if prior_version > 0:
                prior_revision = db.query(UsCodeSectionRevised)\
                    .filter(UsCodeSectionRevised.orig_section_id == section.id)\
                    .filter(UsCodeSectionRevised.version == prior_version)\
                    .first()
                if prior_revision and prior_revision.revised_complexity is not None:
                    current_complexity_score = f"{prior_revision.revised_complexity:.2f}"
            else:
                # For version 0, we might want to get the original complexity if it exists
                # This depends on your data model - adjust if needed
                impact = impact or db.query(SectionImpact).filter(SectionImpact.section_id == section.id).first()
                if impact and hasattr(impact, 'complexity_score') and impact.complexity_score is not None:
                    current_complexity_score = f"{impact.complexity_score:.2f}"

            # Get related exemptions only for version 0 (original text)
            related_exemptions_str = "N/A"
            if prior_version == 0:
                exemptions = db.query(Exemption)\
                    .filter(Exemption.section_id == section.id)\
                    .all()
                
                # Format exemptions information
                if exemptions:
                    exemptions_info = []
                    for e in exemptions:
                        impact = f"${e.revenue_impact_estimate/1000:.1f}M" if e.revenue_impact_estimate else "N/A"
                        entities = f"{e.entity_impact:,.0f}" if e.entity_impact else "N/A"
                        text = e.relevant_text[:100] + "..." if e.relevant_text and len(e.relevant_text) > 100 else e.relevant_text or "N/A"
                        exemptions_info.append(f"Text: {text}, Impact: {impact}, Entities: {entities}")
                    related_exemptions_str = "\n- " + "\n- ".join(exemptions_info)

            # 3. Construct Agent Input String
            relevant_text_input = f"""
            Section ID: {section.section_number} (DB ID: {section.id})
            Original Text (Version {prior_version}): {input_text}
            Complexity Score (Current): {current_complexity_score}
            Revenue Impact ($M): {revenue_impact_str}
            Related Exemptions: {related_exemptions_str}
            """

            # 4. Invoke Agent
            logger.debug(f"Invoking agent with input:\\n{relevant_text_input}")
            response = agent_executor.invoke({"relevant_text": relevant_text_input})
            agent_output = response.get('output')
            logger.debug(f"Agent raw output: {agent_output}")

            if not agent_output:
                logger.error(f"Agent did not return 'output' for section {section.id}. Skipping.")
                continue

            # 5. Parse Agent Output
            parsed_output: Optional[TaxEditorOutput] = None
            try:
                # Use regex to find the first JSON object within the output string
                agent_output_str = str(agent_output).strip()
                # Use non-greedy matching (.*?) to find the *shortest* block
                match = re.search(r'\{.*?\}', agent_output_str, re.DOTALL)

                if not match:
                    raise ValueError("No valid JSON object found in agent output.")

                potential_json_str = match.group(0)

                # First, try parsing the string with standard json library
                try:
                    data_dict = json.loads(potential_json_str)
                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to decode extracted JSON string for section {section.id}: {json_e}")
                    logger.error(f"Extracted string was: {potential_json_str}")
                    logger.error(f"Original raw output was: {agent_output}")
                    continue # Skip if basic JSON parsing fails

                # If JSON string is valid, validate with Pydantic model
                try:
                    parsed_output = TaxEditorOutput.model_validate(data_dict)
                    logger.info(f"Agent decision for section {section.id}: {parsed_output.action}")
                    logger.debug(f"Agent parsed result data: {parsed_output.model_dump()}")
                except Exception as pydantic_e: # Catch Pydantic validation errors specifically
                    logger.error(f"Pydantic validation failed for section {section.id}: {pydantic_e}")
                    logger.error(f"Data dictionary passed to Pydantic was: {data_dict}")
                    logger.error(f"Original raw output was: {agent_output}")
                    continue # Skip if Pydantic validation fails

            except ValueError as ve:
                logger.error(f"JSON extraction failed for section {section.id}: {ve}")
                logger.error(f"Original raw output was: {agent_output}")
                continue
            except Exception as e: # Catch other unexpected errors during extraction/parsing
                 logger.error(f"An unexpected error occurred during parsing/extraction for section {section.id}: {e}", exc_info=True)
                 logger.error(f"Original raw output was: {agent_output}")
                 continue

            # Ensure parsed_output is not None before proceeding
            if parsed_output is None:
                logger.error(f"Skipping section {section.id} due to parsing/validation failure.")
                continue

            # 6. Persist Results to SectionVersion (within the loop's session)
            action = parsed_output.action
            is_deleted = (action == 'delete')
            revised_text = parsed_output.after_text # Already Optional[str]

            new_version_record = UsCodeSectionRevised(
                orig_section_id=section.id,
                version=working_version,
                deleted=is_deleted,
                core_text=revised_text, # Use directly from parsed output
                # Use the parsed float value directly, None if not present
                revised_complexity = parsed_output.new_complexity_score, 
                # Use the parsed Decimal values directly, handle None
                revised_financial_impact = (
                    parsed_output.estimated_kept_dollars if not is_deleted and parsed_output.estimated_kept_dollars is not None
                    else parsed_output.estimated_deleted_dollars * -1 if is_deleted and parsed_output.estimated_deleted_dollars is not None
                    else None # Explicitly handle case where neither is set or applicable
                ),
                # Replicate other relevant fields from original UsCodeSection
                subtitle=section.subtitle,
                chapter=section.chapter,
                subchapter=section.subchapter,
                part=section.part,
                section_number=section.section_number,
                section_title=section.section_title,
            )

            # --- Optional: Create SectionHistory record ---
            history_record = SectionHistory(
                 orig_section_id=section.id,
                 version_changed=working_version,
                 action=action, # Use action from parsed output
                 rationale=parsed_output.rationale # Use rationale from parsed output
            )
            db.add(history_record) # Add history record to the session
            # -------------------------------------------

            db.add(new_version_record)
            db.commit()
            logger.info(f"Successfully saved edited version {working_version} for section {section.id}.")
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed processing section {section.id}: {e}", exc_info=True)
            if db: # Rollback if session exists
                try:
                    db.rollback()
                    logger.warning(f"Rolled back changes for section {section.id} due to error.")
                except Exception as rb_e:
                     logger.error(f"Error during rollback for section {section.id}: {rb_e}")
            # Continue to the next section
            continue
        finally:
            # Ensure session is closed for this section
            if db:
                db.close()
                logger.debug(f"Closed session for section {section.id}")

    logger.info(f"--- Run Finished ---")
    logger.info(f"Successfully processed {processed_count} sections.")
    if limit and processed_count == limit:
         logger.info(f"Reached processing limit of {limit}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Tax Editor Agent to process sections.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sections to process in this run."
    )
    parser.add_argument(
        "--clear",
        action='store_true',
        help="Clear existing edits for the current working version before starting."
    )
    parser.add_argument(
        "--quiet",
        action='store_true',
        help="Suppress INFO and DEBUG log messages, showing only WARNINGs and errors."
    )

    args = parser.parse_args()

    logger.info("Starting Tax Editor Agent run...")
    logger.info(f"Arguments: limit={args.limit}, clear={args.clear}, quiet={args.quiet}")

    # Adjust logging level if --quiet is specified
    if args.quiet:
        print("Quiet mode enabled. Suppressing INFO and DEBUG messages.") # Still print this one confirmation
        logging.getLogger().setLevel(logging.WARNING) # Set root logger level

    process_sections(limit=args.limit, clear=args.clear)

    logger.info("Tax Editor Agent run complete.") 