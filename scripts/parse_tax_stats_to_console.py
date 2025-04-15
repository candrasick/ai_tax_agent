#!/usr/bin/env python
"""
Script to parse tax statistics JSON files from a directory, filter the results,
and print the combined, aggregated results to the console as JSON.

Filters to show only line items present in at least two different unit types.
"""

import os
import sys
import argparse
import json
import logging
from typing import List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Import the parser and the Pydantic model
    from ai_tax_agent.parsers.json_utils import parse_tax_stats_json, CombinedFormLineItem 
except ImportError as e:
    logging.error(f"Failed to import parser function/model: {e}")
    logging.error("Ensure ai_tax_agent/parsers/json_utils.py exists and is importable.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Parse tax statistics JSON files and print combined results, filtered.")
    parser.add_argument("--stats-dir", required=True, 
                        help="Path to the directory containing tax statistics JSON files (e.g., data/tax_statistics).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the parser.")
    args = parser.parse_args()

    # Set logger level for the parser module if different from root
    parser_logger = logging.getLogger('ai_tax_agent.parsers.json_utils')
    parser_logger.setLevel(getattr(logging, args.log_level))

    if not os.path.isdir(args.stats_dir):
        logging.error(f"Specified directory does not exist: {args.stats_dir}")
        sys.exit(1)

    logging.info(f"Starting parsing of directory: {args.stats_dir}")
    combined_data: List[CombinedFormLineItem] = parse_tax_stats_json(args.stats_dir)

    if not combined_data:
        logging.warning("Parsing completed, but no combined data was generated.")
        print("[]") # Output empty JSON list
        sys.exit(0)

    logging.info(f"Successfully parsed and combined {len(combined_data)} line items initially.")
    
    # --- Filter the results --- 
    filtered_data = []
    for item in combined_data:
        unit_count = 0
        if item.dollars is not None: unit_count += 1
        if item.individuals is not None: unit_count += 1
        if item.forms is not None: unit_count += 1
        
        if unit_count >= 2:
            filtered_data.append(item)
    # ------------------------
    
    logging.info(f"Filtered down to {len(filtered_data)} items present in at least 2 unit types.")
    
    if not filtered_data:
        logging.warning("Filtering removed all items. No items found in at least 2 unit types.")
        print("[]") # Output empty JSON list
        sys.exit(0)

    # Convert Filtered Pydantic objects to JSON serializable list of dicts
    try:
        # Use the filtered list here
        output_list = [item.model_dump(exclude_none=True) for item in filtered_data] 
    except Exception as e:
        logging.error(f"Error converting filtered Pydantic objects to dictionaries: {e}")
        sys.exit(1)

    # Print the filtered results as formatted JSON to stdout
    print(json.dumps(output_list, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 