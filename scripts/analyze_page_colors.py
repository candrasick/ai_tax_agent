"""
Test script for the LLM-based blue amount extraction function.

Takes a PDF path and page number, calls `extract_blue_amounts_llm`,
and prints the returned list of amounts and positions.
"""

import argparse
import pdfplumber
import os
import logging
import sys
import json # Need json for pretty printing

# Add project root to path to allow sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy pdfminer logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Import the specific function to test
try:
    from ai_tax_agent.parsers.pdf_parser_utils import extract_blue_amounts_llm
except ImportError as e:
    logging.error(f"Failed to import required function from pdf_parser_utils: {e}")
    logging.error("Ensure the file exists and there are no circular dependencies.")
    sys.exit(1)

def test_llm_blue_amount_extraction(pdf_path: str, page_num: int):
    """Opens a PDF page and calls the LLM blue amount extractor."""
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                logging.error(f"Page number {page_num} is out of range (1-{len(pdf.pages)}).")
                return

            page = pdf.pages[page_num - 1] # pdfplumber is 0-indexed
            logging.info(f"Testing extract_blue_amounts_llm for page {page_num} of '{os.path.basename(pdf_path)}'...")

            # Call the function to test
            result = extract_blue_amounts_llm(page)

            # Print results
            print(f"\n--- Result from extract_blue_amounts_llm (Page {page_num}) ---")
            if result:
                # Pretty print the JSON output
                print(json.dumps(result, indent=2))
            elif isinstance(result, list) and not result:
                 print("Function returned an empty list (no blue numeric amounts found or kept after filtering)." )
            else:
                print(f"Function returned unexpected type or None: {type(result)}")
            print("---------------------------------------------------")

    except Exception as e:
        logging.error(f"An error occurred during PDF processing or LLM call: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Test the LLM-based blue amount extraction for a specific PDF page.")
    parser.add_argument("--pdf-path", required=True, help="Path to the input PDF file (e.g., data/tax_statistics/corporations.pdf).")
    parser.add_argument("--page-num", type=int, required=True, help="The 1-based page number to test.")
    args = parser.parse_args()

    test_llm_blue_amount_extraction(args.pdf_path, args.page_num)

if __name__ == "__main__":
    main() 