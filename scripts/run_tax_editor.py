# scripts/run_tax_editor.py
import logging
import argparse
import json
import os
import sys
from decimal import Decimal
from typing import Optional
import warnings

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
from ai_tax_agent.database.models import UsCodeSection, UsCodeSectionRevised, SectionImpact, SectionHistory
from ai_tax_agent.database.versioning import determine_version_numbers

# Agent imports
from ai_tax_agent.agents import create_tax_editor_agent
from ai_tax_agent.settings import settings # May need settings for agent config

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
    db: Session | None = None
    try:
        db = get_session()
        if not db:
            logger.error("Failed to get database session.")
            return

        # Call determine_version_numbers without the db argument
        prior_version, working_version = determine_version_numbers()
        logger.info(f"Determined versions: Prior={prior_version}, Working={working_version}")

        if working_version == 0:
            logger.error("Working version is 0. Cannot run editor agent without an initial version.")
            return

        if clear:
            clear_working_version_edits(db, working_version)
            # Re-determine versions in case clearing affected them (unlikely but safe)
            prior_version, working_version = determine_version_numbers()

        # --- Create Agent ---
        logger.info("Creating Tax Editor Agent...")
        # Set verbose=False to reduce agent output
        agent_executor = create_tax_editor_agent(llm_model_name = "gemini-1.5-pro", verbose=False)
        if not agent_executor:
            logger.error("Failed to create Tax Editor Agent.")
            return
        logger.info("Tax Editor Agent created successfully.")

        # --- Identify Sections to Process ---
        # Find sections that DO NOT have a SectionVersion for the working_version
        subquery = db.query(UsCodeSectionRevised.orig_section_id).filter(UsCodeSectionRevised.version == working_version)
        query = db.query(UsCodeSection).filter(UsCodeSection.id.notin_(subquery))

        # Optional: Add ordering if needed, e.g., by section number
        query = query.order_by(UsCodeSection.section_number) # Adjust field name if necessary

        if limit is not None and limit > 0:
            query = query.limit(limit)

        sections_to_process = query.all()

        if not sections_to_process:
            logger.info(f"No sections found needing processing for version {working_version}.")
            return

        logger.info(f"Found {len(sections_to_process)} sections to process (limit={limit}).")

        # --- Processing Loop ---
        processed_count = 0
        # Disable tqdm progress bar if quiet mode is enabled
        # Need to check if 'args' exists in locals() in case process_sections is called programmatically elsewhere
        disable_tqdm = locals().get('args') and locals().get('args').quiet
        # Wrap the loop with tqdm for a progress bar
        for section in tqdm(sections_to_process, desc="Processing Sections", unit="section", disable=disable_tqdm):
            # Reduce logging noise inside the loop, especially in non-quiet mode
            # Keep minimal INFO log even in normal mode, suppress in quiet
            if not disable_tqdm:
                 logger.info(f"Processing Section ID: {section.id} ({section.section_number})")
            try:
                # 1. Get Input Text (from prior version)
                # Fetch text directly based on prior_version
                input_text: Optional[str] = None # Add type hint
                if prior_version == 0:
                    # Get text from original UsCodeSection table
                    original_section = db.query(UsCodeSection.core_text).filter(UsCodeSection.id == section.id).scalar()
                    input_text = original_section
                    logger.debug(f"Fetched original text (v0) for section {section.id}. Length: {len(input_text) if input_text else 0}")
                elif prior_version > 0:
                    # Get text from UsCodeSectionRevised table for the prior version
                    revised_section_text = db.query(UsCodeSectionRevised.core_text)\
                                             .filter(UsCodeSectionRevised.orig_section_id == section.id,
                                                     UsCodeSectionRevised.version == prior_version)\
                                             .scalar()
                    input_text = revised_section_text
                    logger.debug(f"Fetched revised text (v{prior_version}) for section {section.id}. Length: {len(input_text) if input_text else 0}")

                if input_text is None:
                     logger.warning(f"Could not find text for prior version {prior_version} for section {section.id}. Skipping.")
                     continue

                # 2. Get Additional Context (Simulate based on sample, enhance as needed)
                # Query impact data - assuming one impact record per section for simplicity
                impact = db.query(SectionImpact).filter(SectionImpact.section_id == section.id).first()
                revenue_impact_str = f"{impact.revenue_impact / 1_000_000:.1f}" if impact and impact.revenue_impact is not None else "N/A"
                # TODO: Get real complexity score (perhaps from prior SectionVersion?) and exemptions
                current_complexity_score = "N/A" # Placeholder
                related_exemptions = "N/A"      # Placeholder

                # 3. Construct Agent Input String
                relevant_text_input = f"""
                Section ID: {section.section_number} (DB ID: {section.id})
                Original Text (Version {prior_version}): {input_text}
                Complexity Score (Current): {current_complexity_score}
                Revenue Impact ($M): {revenue_impact_str}
                Related Exemptions: {related_exemptions}
                """

                # 4. Invoke Agent
                logger.debug(f"Invoking agent with input:\n{relevant_text_input}")
                response = agent_executor.invoke({"relevant_text": relevant_text_input})
                agent_output = response.get('output')
                logger.debug(f"Agent raw output: {agent_output}")

                if not agent_output:
                    logger.error(f"Agent did not return 'output' for section {section.id}. Skipping.")
                    continue

                # 5. Parse Agent Output JSON
                try:
                    # The agent should output JUST the JSON string
                    # Sometimes agents add backticks or "json" prefix
                    if agent_output.strip().startswith("```json"):
                         agent_output = agent_output.strip()[7:-3].strip()
                    elif agent_output.strip().startswith("```"):
                         agent_output = agent_output.strip()[3:-3].strip()

                    result_data = json.loads(agent_output)
                    logger.info(f"Agent decision for section {section.id}: {result_data.get('action')}")
                    logger.debug(f"Agent result data: {result_data}")

                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to parse agent JSON output for section {section.id}: {json_e}")
                    logger.error(f"Raw output was: {agent_output}")
                    continue # Skip persisting if output is bad

                # 6. Persist Results to SectionVersion
                # **UPDATED**: Use UsCodeSectionRevised and its fields
                # Check UsCodeSectionRevised fields: orig_section_id, version, deleted, core_text,
                # revised_complexity, revised_financial_impact
                # We need to store action/rationale elsewhere (maybe SectionHistory or add fields?)
                # For now, store what we can in UsCodeSectionRevised
                action = result_data.get('action')
                is_deleted = (action == 'delete')
                # Store the agent's generated text if simplified/redrafted, otherwise store None
                revised_text = result_data.get('after_text') if action in ['simplify', 'redraft'] else None

                new_version_record = UsCodeSectionRevised(
                    orig_section_id=section.id,
                    version=working_version,
                    deleted=is_deleted,
                    core_text=revised_text, # Store the new text or None if deleted/kept
                    revised_complexity = float(result_data.get('new_complexity_score')) if result_data.get('new_complexity_score') is not None else None,
                    # Store estimated financial impact for this version.
                    # Use kept_dollars if present and not deleted, else use deleted_dollars if present, else None.
                    revised_financial_impact = (
                        Decimal(str(result_data.get('estimated_kept_dollars'))) if result_data.get('estimated_kept_dollars') is not None and not is_deleted
                        else Decimal(str(result_data.get('estimated_deleted_dollars'))) * -1 if result_data.get('estimated_deleted_dollars') is not None and is_deleted # Store deleted as negative impact
                        else None
                    ),
                    # --- TODO: Store action/rationale --- 
                    # This might require adding fields to UsCodeSectionRevised or creating a SectionHistory entry

                    # Replicate other relevant fields from original UsCodeSection
                    subtitle=section.subtitle,
                    chapter=section.chapter,
                    subchapter=section.subchapter,
                    part=section.part,
                    section_number=section.section_number,
                    section_title=section.section_title,
                )

                # --- Optional: Create SectionHistory record ---
                # Create SectionHistory record
                history_record = SectionHistory(
                     orig_section_id=section.id,
                     version_changed=working_version, # Record the version where the change happened
                     action=action, # Use the 'action' variable derived from agent output
                     rationale=result_data.get('rationale') # Get rationale from agent output
                )
                db.add(history_record) # Add history record to the session
                # -------------------------------------------

                db.add(new_version_record)
                db.commit()
                logger.info(f"Successfully saved edited version {working_version} for section {section.id}.")
                processed_count += 1

            except Exception as e:
                db.rollback() # Rollback changes for this specific section on error
                logger.error(f"Failed processing section {section.id}: {e}", exc_info=True)
                # Decide whether to stop or continue processing others
                # For now, let's continue
                continue

        logger.info(f"--- Run Finished ---")
        logger.info(f"Successfully processed {processed_count} sections.")
        if limit and processed_count == limit:
             logger.info(f"Reached processing limit of {limit}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if db:
            db.rollback() # Rollback any potential dangling transaction
    finally:
        if db:
            db.close()
            logger.debug("Database session closed.")


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