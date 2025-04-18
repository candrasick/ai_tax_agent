#!/usr/bin/env python
"""Analyzes a single US Code section for financial/entity impact using linked form stats."""

import argparse
import json
import logging
import sys
from decimal import Decimal
from typing import Dict, Any, List, Optional # Updated imports
from pathlib import Path
import re # <-- Import re module
from sqlalchemy.orm import Session
from tqdm import tqdm

# Adjust import path based on your project structure
# This assumes the script is run from the project root directory
sys.path.append('.') # Add project root to path if necessary

from ai_tax_agent.tools.db_tools import get_section_details_and_stats
from ai_tax_agent.database.session import get_session # Needed for potential future DB writes
from ai_tax_agent.agents import create_tax_analysis_agent # <-- Import the agent creator
from langchain_core.runnables import Runnable # For type hinting agent executor
from ai_tax_agent.database.models import UsCodeSection, SectionImpact, Exemption

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper Function --- 

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Attempts to parse a JSON object from the agent's response string,
       handling potential markdown code fences."""
    # Regex to find JSON block within ```json ... ``` fences
    # DOTALL flag allows . to match newlines
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    
    json_string = None
    if match:
        json_string = match.group(1)
        logger.debug("Extracted JSON string from markdown fences.")
    else:
        # Fallback: If no fences, try finding the first '{' and last '}'
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = response_text[start_index : end_index + 1]
            logger.debug("Extracted JSON string using brace finding fallback.")
        else:
            logger.warning("Could not find JSON structure in response.")
            return None # No JSON structure found

    if json_string:
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extracted JSON string: {e}. String: {json_string[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}")
            return None
    else:
        # This case should theoretically not be reached if the above logic is sound
        logger.warning("JSON string was None after extraction attempts.")
        return None

# --- Core Analysis Functions --- 

def analyze_single_exemption(
    llm_chain: Runnable, 
    exemption_prompt_template: str, 
    exemption_text: str, 
    section_context: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze an individual exemption using the LLM chain and return structured results."""
    if not exemption_text:
        return {
            "estimated_revenue_impact": None,
            "estimated_entity_impact": None,
            "rationale": "No exemption text provided."
        }

    try:
        # Format the prompt using the loaded template and provided context
        full_prompt = exemption_prompt_template.format(
            relevant_text=exemption_text,
            # section_context=section_context or "No additional section context provided."
            # Note: The current prompt file doesn't have a section_context placeholder.
            # If you add it to the prompt file, uncomment the line above.
        )
    except KeyError as e:
        logger.error(f"Failed to format prompt template: Missing key {e}")
        return {
            "estimated_revenue_impact": None,
            "estimated_entity_impact": None,
            "rationale": f"Error: Could not format prompt template (Missing key: {e})."
        }

    logger.debug(f"Analyzing exemption with prompt: {full_prompt[:300]}...")
    try:
        agent_response = llm_chain.invoke({"input": full_prompt})
        agent_output_str = agent_response.get('output', '')
        logger.debug(f"Agent Raw Output: {agent_output_str}")
        
        parsed_response = extract_json_from_response(agent_output_str)

        if parsed_response:
             # Ensure expected keys exist, default to None if missing
             return {
                "estimated_revenue_impact": parsed_response.get('estimated_revenue_impact'),
                "estimated_entity_impact": parsed_response.get('estimated_entity_impact'),
                "rationale": parsed_response.get('rationale', 'Agent returned JSON but no rationale.')
             }
        else:
            # Use raw output as rationale if JSON parsing failed
            return {
                "estimated_revenue_impact": None,
                "estimated_entity_impact": None,
                "rationale": f"Failed to parse LLM JSON output. Raw response: {agent_output_str}"
            }

    except Exception as e:
        logger.error(f"Error invoking agent or processing output for exemption: {e}", exc_info=True)
        return {
            "estimated_revenue_impact": None,
            "estimated_entity_impact": None,
            "rationale": f"Exception during agent analysis: {str(e)}"
        }

def analyze_section_exemptions(
    llm_chain: Runnable, 
    exemption_prompt_template: str, 
    section_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Analyze all exemptions linked to a section using the LLM chain."""
    # Use core_text as context, truncate if needed
    section_context = (section_data.get("core_text") or "")[:1000] 
    exemption_results = []
    original_exemptions = section_data.get("exemptions", [])
    logger.info(f"Analyzing {len(original_exemptions)} exemptions for section {section_data.get('section_id')}...")

    for exemption in original_exemptions:
        exemption_id = exemption.get("exemption_id")
        relevant_text = exemption.get("relevant_text", "").strip()
        logger.debug(f"Processing exemption ID: {exemption_id}")

        analysis_result = analyze_single_exemption(
            llm_chain,
            exemption_prompt_template, 
            relevant_text, 
            section_context # Pass section context (if prompt template supports it)
        )
        
        # Combine analysis result with original exemption info
        combined_result = {
            "exemption_id": exemption_id,
            "revenue_impact_estimate": analysis_result.get("estimated_revenue_impact"),
            "entity_impact": analysis_result.get("estimated_entity_impact"),
            "analysis_rationale": analysis_result.get("rationale"),
            "original_relevant_text": relevant_text,
            "original_rationale": exemption.get("original_rationale") # Use original_rationale key from db_tools output
        }
        exemption_results.append(combined_result)

    return exemption_results

def analyze_entire_section(
    llm_chain: Runnable, 
    exemption_prompt_template: str, 
    section_identifier: str
) -> Dict[str, Any]:
    """Analyze a full tax code section for revenue and entity impact estimates."""
    logger.info(f"Retrieving data for section identifier: {section_identifier}")
    section_data = get_section_details_and_stats(section_identifier)

    if isinstance(section_data, str): # Handle error case from db_tool
        logger.error(f"Failed to retrieve data for section '{section_identifier}': {section_data}")
        raise ValueError(f"Failed to retrieve data for section '{section_identifier}': {section_data}")
    
    if not section_data:
        logger.error(f"No data returned for section '{section_identifier}'.")
        raise ValueError(f"No data returned for section '{section_identifier}'.")

    # Section-level summary from precomputed stats in the retrieved data
    section_id = section_data.get("section_id")
    summary = section_data.get("aggregation_summary", {})
    total_dollars = summary.get("total_dollars", 0.0)
    total_forms = summary.get("total_forms", 0.0)
    total_people = summary.get("total_people", 0.0)

    # Calculate direct impact based on aggregated stats
    revenue_impact_direct = float(total_dollars)
    entity_impact_direct = float(total_people) if total_people > 0 else float(total_forms)
    direct_rationale = f"Based on aggregation of {summary.get('linked_field_count', 0)} form field(s) statistically linked to section {section_data.get('section_identifier')} ({section_id})."

    section_summary = {
        "section_identifier": section_data.get("section_identifier"),
        "section_id": section_id,
        "section_title": section_data.get("section_title"), # Fetched by updated db_tool
        "core_text_snippet": (section_data.get("core_text") or "")[:500] + ("..." if len(section_data.get("core_text") or "") > 500 else ""),
        "direct_revenue_impact": revenue_impact_direct,
        "direct_entity_impact": entity_impact_direct,
        "direct_impact_rationale": direct_rationale,
        "aggregation_details": summary # Include the raw aggregation numbers
    }

    # Analyze exemptions using the LLM chain
    exemptions_analysis = analyze_section_exemptions(llm_chain, exemption_prompt_template, section_data)

    return {
        "section_summary": section_summary,
        "exemptions_analysis": exemptions_analysis
    }

# --- Database Saving Function ---
def save_analysis_results(db: Session, section_summary: Dict[str, Any], exemptions_analysis: List[Dict[str, Any]]) -> tuple[bool, bool]: # Added return type hint
    """Saves the analysis results to the database.

    Returns:
        A tuple: (bool indicating if save was successful, bool indicating if impact was null).
    """
    section_id = section_summary.get("section_id")
    if not section_id:
        logger.error("Section summary missing section_id. Cannot save.")
        return False, True # Indicate save failure and treat as null impact

    results_saved = False
    null_results_section = True # Track if section impact is effectively null
    null_results_exemptions = True # Track if all exemption impacts are null

    try:
        # --- Save/Update SectionImpact ---
        impact_record = db.query(SectionImpact).filter(SectionImpact.section_id == section_id).first()
        if not impact_record:
            impact_record = SectionImpact(section_id=section_id)
            db.add(impact_record)
            logger.debug(f"Creating new SectionImpact record for section_id: {section_id}")
        else:
            logger.debug(f"Updating existing SectionImpact record for section_id: {section_id}")

        # Convert potentially None values from analysis to Decimal or keep as None for DB
        # Check types robustly before conversion
        revenue_impact_val = section_summary.get('direct_revenue_impact')
        entity_impact_val = section_summary.get('direct_entity_impact')
        
        revenue_impact = Decimal(revenue_impact_val) if isinstance(revenue_impact_val, (int, float)) else None
        entity_impact = Decimal(entity_impact_val) if isinstance(entity_impact_val, (int, float)) else None

        impact_record.revenue_impact = revenue_impact
        impact_record.entity_impact = entity_impact
        impact_record.rationale = section_summary.get('direct_impact_rationale')

        if revenue_impact is not None or entity_impact is not None:
             null_results_section = False

        # --- Update Exemptions --- Add Exemption import
        for exemption_result in exemptions_analysis:
            exemption_id = exemption_result.get("exemption_id")
            if not exemption_id:
                logger.warning("Exemption analysis result missing exemption_id. Skipping update.")
                continue

            exemption_record = db.query(Exemption).filter(Exemption.id == exemption_id).first()
            if not exemption_record:
                logger.warning(f"Could not find Exemption record with id: {exemption_id} to update.")
                continue

            # Convert potentially None values from analysis to Decimal or keep as None for DB
            rev_impact_est_val = exemption_result.get('revenue_impact_estimate')
            ent_impact_est_val = exemption_result.get('entity_impact')
            rev_impact_est = None
            ent_impact_est = None

            try:
                 if isinstance(rev_impact_est_val, (int, float)):
                      rev_impact_est = Decimal(rev_impact_est_val)
                 elif isinstance(rev_impact_est_val, str):
                      # Attempt conversion if it looks like a number, otherwise None
                      rev_impact_est = Decimal(rev_impact_est_val.replace(",","")) # Handle commas
            except (ValueError, TypeError, InvalidOperation):
                 logger.warning(f"Invalid format for revenue_impact_estimate on exemption {exemption_id}: '{rev_impact_est_val}'. Setting to None.")
                 rev_impact_est = None

            try:
                 if isinstance(ent_impact_est_val, (int, float)):
                      ent_impact_est = Decimal(ent_impact_est_val)
                 elif isinstance(ent_impact_est_val, str):
                      ent_impact_est = Decimal(ent_impact_est_val.replace(",","")) # Handle commas
            except (ValueError, TypeError, InvalidOperation):
                 logger.warning(f"Invalid format for entity_impact on exemption {exemption_id}: '{ent_impact_est_val}'. Setting to None.")
                 ent_impact_est = None

            exemption_record.revenue_impact_estimate = rev_impact_est
            exemption_record.entity_impact = ent_impact_est
            exemption_record.rationale = exemption_result.get('analysis_rationale')

            if rev_impact_est is not None or ent_impact_est is not None:
                 null_results_exemptions = False

            logger.debug(f"Updated Exemption record id: {exemption_id}")

        db.commit()
        logger.info(f"Successfully saved analysis results for section_id: {section_id}")
        results_saved = True

    except Exception as e:
        logger.error(f"Database error saving results for section_id {section_id}: {e}", exc_info=True)
        db.rollback() # Rollback on error
        results_saved = False # Explicitly mark as not saved

    # Determine if the overall result was null
    is_null_result = null_results_section and null_results_exemptions
    return results_saved, is_null_result

# --- Main Execution --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the financial and entity impact of US Code sections using linked stats and an agent for exemptions.")
    # Remove positional argument, add flags
    # parser.add_argument("section_identifier", type=str, help="The US Code section number (e.g., '162') or its database ID.") # REMOVE THIS LINE
    parser.add_argument("--clear", action="store_true", help="Re-analyze all sections, ignoring existing results in section_impact table.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sections to analyze.")
    # Add flag to specify section ID directly (optional)
    parser.add_argument("--section", type=str, default=None, help="Analyze only a specific section identifier (number or ID).")

    args = parser.parse_args()

    # Load Prompt Template Once
    prompt_file_path = Path("prompts/analyze_exemption_impact.txt")
    try:
        exemption_prompt_template_content = prompt_file_path.read_text()
        logger.info(f"Loaded exemption analysis prompt from {prompt_file_path}")
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {prompt_file_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading prompt file {prompt_file_path}: {e}. Exiting.")
        sys.exit(1)

    # Initialize LLM Chain (Agent Executor) Once
    logger.info("Initializing Tax Analysis Agent (LLM Chain)...")
    llm_chain_executor = create_tax_analysis_agent(verbose=False) # verbose=True for debugging agent steps
    if not llm_chain_executor:
        logger.error("Failed to initialize the Tax Analysis Agent. Exiting.")
        sys.exit(1)
    logger.info("Tax Analysis Agent (LLM Chain) initialized successfully.")

    # --- Determine Sections to Process ---
    sections_to_process = []
    
    # --- Get DB Session --- 
    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session. Exiting.")
        sys.exit(1)
    # ---------------------

    try: # Use try...finally to ensure db is closed
        if args.section:
             # If a specific section is provided, analyze only that one
             sections_to_process = [args.section] # Keep as string identifier
             logger.info(f"Analyzing specific section identifier provided: {args.section}")
        else:
            logger.info("Fetching sections from database...")
            # Now db should be defined
            all_section_ids_query = db.query(UsCodeSection.id) 
            all_section_ids = [s.id for s in all_section_ids_query.all()]
            logger.info(f"Found {len(all_section_ids)} total sections.")

            if not args.clear:
                logger.info("Checking for existing impact data (running in incremental mode)...")
                existing_impact_ids_query = db.query(SectionImpact.section_id)
                existing_impact_ids = {s.section_id for s in existing_impact_ids_query.all()}
                logger.info(f"Found {len(existing_impact_ids)} sections with existing impact data.")
                sections_to_process_ids = [sid for sid in all_section_ids if sid not in existing_impact_ids]
                logger.info(f"{len(sections_to_process_ids)} sections remaining to be analyzed.")
            else:
                logger.info("Running in --clear mode. Analyzing all sections.")
                sections_to_process_ids = all_section_ids

            # Apply limit if specified (after filtering)
            if args.limit is not None and args.limit > 0:
                logger.info(f"Applying limit: Analyzing only the first {args.limit} sections.")
                sections_to_process_ids = sections_to_process_ids[:args.limit]

            # Convert IDs back to strings for the analysis function if needed, or pass IDs
            sections_to_process = [str(sid) for sid in sections_to_process_ids]


        if not sections_to_process:
            logger.info("No sections found to process based on current criteria.")
            # No need to exit here, just let the loop below handle the empty list
            # sys.exit(0) # Remove this exit

        logger.info(f"Starting analysis for {len(sections_to_process)} sections...")

        # --- Analysis Loop ---
        summary_stats = {
            "processed": 0,
            "saved": 0,
            "no_impact": 0, # Sections where analysis returned all nulls
            "errors": 0
        }

        # Use tqdm for progress bar
        for section_id_or_num in tqdm(sections_to_process, desc="Analyzing Sections"):
            try:
                analysis_result = analyze_entire_section(
                    llm_chain_executor,
                    exemption_prompt_template_content,
                    section_id_or_num # Pass identifier
                )

                if analysis_result:
                     saved, is_null = save_analysis_results(
                         db,
                         analysis_result["section_summary"],
                         analysis_result["exemptions_analysis"]
                     )
                     summary_stats["processed"] += 1
                     if saved:
                         summary_stats["saved"] += 1
                         if is_null:
                             summary_stats["no_impact"] += 1
                     else:
                         summary_stats["errors"] += 1
                else:
                     # Should not happen if analyze_entire_section raises ValueError
                     logger.warning(f"analyze_entire_section returned None for {section_id_or_num}, expected exception.")
                     summary_stats["errors"] += 1


            except ValueError as ve: # Catch errors from analyze_entire_section (e.g., DB fetch fail)
                logger.error(f"Analysis failed for section {section_id_or_num}: {ve}")
                summary_stats["errors"] += 1
            except Exception as e: # Catch unexpected errors during the loop
                logger.error(f"Unexpected error processing section {section_id_or_num}: {e}", exc_info=True)
                summary_stats["errors"] += 1
                # Optional: Decide whether to break the loop on unexpected errors
                # break

    finally:
        # --- Cleanup and Summary ---
        if db: # Ensure db exists before trying to close
            db.close()
            logger.debug("Database session closed.")
            
        logger.info("--- Analysis Complete ---")
        logger.info(f"Total sections targeted: {len(sections_to_process)}")
        logger.info(f"Sections processed: {summary_stats['processed']}")
        logger.info(f"Results saved successfully: {summary_stats['saved']}")
        logger.info(f"Sections with no discernible impact (all nulls): {summary_stats['no_impact']}")
        logger.info(f"Sections with errors during analysis or saving: {summary_stats['errors']}")
        logger.info("-------------------------") 