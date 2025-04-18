#!/usr/bin/env python
"""Tests the tax simplification agent for a single section."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Adjust import path based on your project structure
sys.path.append('.') 

# Import the new DB tool function and the agent creator
from ai_tax_agent.tools.db_tools import get_section_simplification_context
from ai_tax_agent.agents import create_tax_analysis_agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def format_simplification_input(section_context: Dict[str, Any]) -> str:
    """Formats the fetched context into the string needed for the prompt's relevant_text."""
    
    core_text = section_context.get('core_text', '[CORE TEXT NOT FOUND]')
    complexity = section_context.get('complexity_score', 'N/A')
    sec_rev_impact = section_context.get('section_revenue_impact', 'N/A')
    sec_ent_impact = section_context.get('section_entity_impact', 'N/A')
    exemptions = section_context.get('exemptions', [])

    # Format numbers nicely or show N/A
    complexity_str = f"{complexity:.2f}" if isinstance(complexity, (int, float)) else "N/A"
    sec_rev_impact_str = f"${sec_rev_impact:,.0f}" if isinstance(sec_rev_impact, (int, float)) else "N/A"
    sec_ent_impact_str = f"{sec_ent_impact:,.0f}" if isinstance(sec_ent_impact, (int, float)) else "N/A"

    input_str = f"Section Core Text:\n---\n{core_text}\n---\n\n"
    input_str += f"Section Metrics:\n"
    input_str += f"- Complexity Score: {complexity_str}\n"
    input_str += f"- Financial Impact (Dollars): {sec_rev_impact_str}\n"
    input_str += f"- Entity Impact: {sec_ent_impact_str}\n\n"
    input_str += "Associated Exemptions:\n"

    if exemptions:
        for ex in exemptions:
            ex_id = ex.get('exemption_id')
            ex_text = ex.get('relevant_text', '')
            ex_rev = ex.get('revenue_impact', 'N/A')
            ex_ent = ex.get('entity_impact', 'N/A')
            ex_rev_str = f"${ex_rev:,.0f}" if isinstance(ex_rev, (int, float)) else "N/A"
            ex_ent_str = f"{ex_ent:,.0f}" if isinstance(ex_ent, (int, float)) else "N/A"
            
            input_str += f"---\nExemption ID: {ex_id}\n"
            input_str += f"Exemption Text: {ex_text}\n"
            input_str += f"- Estimated Financial Impact (Dollars): {ex_rev_str}\n"
            input_str += f"- Estimated Entity Impact: {ex_ent_str}\n"
        input_str += "---\n"
    else:
        input_str += "None\n"
        
    return input_str.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the tax simplification agent on a specific section.")
    parser.add_argument("section_identifier", type=str, help="The US Code section number (e.g., '162') or its database ID.")
    parser.add_argument("--prompt", type=str, default="prompts/tax_simplify_prompt.txt", help="Path to the simplification prompt file.")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="Name of the Gemini model to use for the agent.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run agent in verbose mode.")
    args = parser.parse_args()

    # 1. Load Prompt Template
    prompt_file_path = Path(args.prompt)
    try:
        simplification_prompt_template = prompt_file_path.read_text()
        logger.info(f"Loaded simplification prompt from {prompt_file_path}")
    except Exception as e:
        logger.error(f"Error loading prompt file {prompt_file_path}: {e}. Exiting.")
        sys.exit(1)

    # 2. Initialize Agent
    logger.info(f"Initializing Tax Analysis Agent with model: {args.model}...")
    agent_executor = create_tax_analysis_agent(llm_model_name=args.model, verbose=args.verbose)
    if not agent_executor:
        logger.error("Failed to initialize the Tax Analysis Agent. Exiting.")
        sys.exit(1)
    logger.info("Tax Analysis Agent initialized successfully.")

    # 3. Fetch Section Context Data
    logger.info(f"Fetching simplification context for section: {args.section_identifier}...")
    context_data = get_section_simplification_context(args.section_identifier)
    if isinstance(context_data, str): # Handle error string from DB tool
        logger.error(f"Failed to fetch context: {context_data}")
        sys.exit(1)
    logger.info("Successfully fetched context data.")

    # 4. Format the Input for the Prompt
    formatted_relevant_text = format_simplification_input(context_data)
    logger.debug(f"Formatted relevant text for prompt:\n{formatted_relevant_text}")

    # 5. Format the Final Prompt
    try:
        final_prompt = simplification_prompt_template.format(relevant_text=formatted_relevant_text)
    except KeyError:
        logger.error("Failed to format the main prompt template. Ensure '{relevant_text}' placeholder exists.")
        sys.exit(1)
    
    logger.info("Invoking agent with the formatted prompt...")
    # 6. Invoke Agent
    try:
        agent_input = {"input": final_prompt}
        agent_response = agent_executor.invoke(agent_input)
        agent_output = agent_response.get('output', 'Agent did not return output.')
        
        logger.info("Agent execution finished.")
        # 7. Print Agent Output (attempt to pretty-print if JSON)
        print("\n--- Agent Output ---")
        try:
            # Try parsing as JSON for pretty printing
            parsed_json = json.loads(agent_output)
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError:
            # If not JSON, print the raw string output
            print(agent_output)
            
    except Exception as e:
        logger.error(f"Error during agent invocation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Simplification test completed.") 