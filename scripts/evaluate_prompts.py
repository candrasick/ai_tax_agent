"""
Script to evaluate different prompts for multimodal amount association.

Runs the PDF page parser (`parse_pdf_page_structure`) for a specified
PDF page using a list of different prompt template files.

Outputs the resulting JSON structure and summary statistics for each prompt.
"""
import argparse
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Add project root to path to allow sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy pdfminer logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)

try:
    from ai_tax_agent.parsers.pdf_parser_utils import parse_pdf_page_structure, AmountUnit
except ImportError as e:
    logging.error(f"Failed to import required functions from pdf_parser_utils: {e}")
    sys.exit(1)

def calculate_stats(result_data: Dict[str, Any]) -> Dict[str, int]:
    """Calculates assignment stats from the parser result."""
    initial_count = result_data.get("initial_amount_count", 0)
    assigned_count = 0
    if isinstance(result_data.get("line_items"), list):
        assigned_count = sum(1 for item in result_data["line_items"] if item.get("amount") is not None)
    
    unassigned_count = initial_count - assigned_count
    return {
        "initial": initial_count,
        "assigned": assigned_count,
        "unassigned": unassigned_count
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal amount association prompts.")
    parser.add_argument("--pdf-path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--page-num", type=int, required=True, help="The 1-based page number to evaluate.")
    parser.add_argument("--prompt-files", required=True, nargs='+', 
                        help="List of amount ASSOCIATION prompt template filenames (relative to prompts/ dir) to test.")
    parser.add_argument("--blue-amount-prompt", default=None,
                        help="Optional: Specify a specific BLUE AMOUNT EXTRACTION prompt filename (relative to prompts/ dir) to use for all runs.")
    parser.add_argument("--amount-unit", type=lambda s: AmountUnit[s.upper()] if s else None, choices=list(AmountUnit) + [None],
                        default=None, help="Optionally specify the amount unit (dollars, forms, individuals) to override detection.")

    args = parser.parse_args()

    # Validate prompt file paths early
    valid_prompt_files = []
    prompts_base_dir = os.path.join(project_root, 'prompts')
    for prompt_filename in args.prompt_files:
        prompt_file_path = os.path.join(prompts_base_dir, prompt_filename)
        if not os.path.exists(prompt_file_path):
            logging.warning(f"Amount association prompt file not found: {prompt_file_path}. Skipping evaluation for this prompt.")
        else:
            valid_prompt_files.append(prompt_file_path)

    if not valid_prompt_files:
        logging.error("No valid amount association prompt files found to evaluate.")
        sys.exit(1)

    # Validate blue amount prompt path if provided
    blue_prompt_override_path = None
    if args.blue_amount_prompt:
        blue_prompt_override_path = os.path.join(prompts_base_dir, args.blue_amount_prompt)
        if not os.path.exists(blue_prompt_override_path):
            logging.error(f"Blue amount extraction prompt file not found: {blue_prompt_override_path}. Exiting.")
            sys.exit(1)
        logging.info(f"Using specific blue amount extraction prompt: {blue_prompt_override_path}")

    if not os.path.exists(args.pdf_path):
        logging.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)

    for prompt_file_path in valid_prompt_files:
        prompt_filename = os.path.basename(prompt_file_path)

        print(f"\n--- Evaluating Amount Association Prompt: {prompt_filename} ---")
        logging.info(f"Running parser for page {args.page_num} using association prompt: {prompt_file_path}")
        if blue_prompt_override_path:
            logging.info(f"           and using blue amount prompt: {blue_prompt_override_path}")

        try:
            # Call the parser, passing both override paths
            parsed_data = parse_pdf_page_structure(
                pdf_path=args.pdf_path,
                page_num_one_indexed=args.page_num,
                forced_amount_unit=args.amount_unit, # Pass enum member or None
                prompt_template_path_override=prompt_file_path, # Association prompt
                blue_amount_prompt_override=blue_prompt_override_path # Blue amount prompt (or None)
            )

            if parsed_data:
                print("\nParsing Result (JSON):")
                # Remove the initial count before printing JSON if desired for cleaner output
                # output_for_json = {k: v for k, v in parsed_data.items() if k != 'initial_amount_count'}
                # print(json.dumps(output_for_json, indent=2))
                print(json.dumps(parsed_data, indent=2))

                stats = calculate_stats(parsed_data)
                print("\nSummary Statistics:")
                print(f"  - Initial Amounts Detected: {stats['initial']}")
                print(f"  - Amounts Assigned:       {stats['assigned']}")
                print(f"  - Amounts Unassigned:     {stats['unassigned']}")
            else:
                logging.error(f"Parser returned no data for page {args.page_num} using prompt {prompt_filename}.")

        except Exception as e:
            logging.error(f"An error occurred during parsing with prompt {prompt_filename}: {e}", exc_info=True)
        
        print("-----------------------------------------")

    print("\nPrompt evaluation complete.")

if __name__ == "__main__":
    main() 