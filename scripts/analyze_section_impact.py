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

# Adjust import path based on your project structure
# This assumes the script is run from the project root directory
sys.path.append('.') # Add project root to path if necessary

from ai_tax_agent.tools.db_tools import get_section_details_and_stats
from ai_tax_agent.database.session import get_session # Needed for potential future DB writes
from ai_tax_agent.agents import create_tax_analysis_agent # <-- Import the agent creator
from langchain_core.runnables import Runnable # For type hinting agent executor

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

    # Run the main analysis function
    try:
        analysis_result = analyze_entire_section(
            llm_chain_executor, 
            exemption_prompt_template_content, 
            args.section_identifier
        )
        print(json.dumps(analysis_result, indent=2))
        logger.info(f"Successfully completed analysis for section identifier: {args.section_identifier}")
    except ValueError as ve:
        logger.error(f"Analysis failed: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        sys.exit(1) 