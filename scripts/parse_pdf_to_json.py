"""
Script to parse a PDF document page by page using multimodal analysis,
validate essential fields for each page, and output the structured data to a JSON file.
"""

import argparse
import json
import os
import logging
import sys

# Add project root to path to allow sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ai_tax_agent.parsers.pdf_parser_utils import parse_full_pdf_structure, AmountUnit
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy pdfminer warnings (like missing CropBox)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def validate_page_data(page_data: dict) -> bool:
    """Validates if a single page's parsed data contains essential fields."""
    is_valid = True
    page_num = page_data.get('page_number')
    form_title = page_data.get('form_title')
    schedule_title = page_data.get('schedule_title')

    if not page_num:
        logging.warning(f"Validation failed for an entry: Missing 'page_number'. Data: {str(page_data)[:100]}...")
        is_valid = False
        # Assign a placeholder if possible, though the function should add it
        page_num = "UNKNOWN"

    # Check if at least one title is present and not empty/None
    if not (form_title or schedule_title):
        logging.warning(f"Validation failed for page {page_num}: Missing both 'form_title' and 'schedule_title'.")
        is_valid = False

    return is_valid

def main():
    parser = argparse.ArgumentParser(description="Parse a PDF, validate pages, and output to JSON in the same directory.")
    parser.add_argument("--pdf-path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start parsing from (1-based).")
    parser.add_argument("--amount-unit", type=lambda s: AmountUnit[s.upper()], choices=list(AmountUnit),
                        help="Specify the unit for amounts (dollars, forms, individuals). Overrides detection.")

    args = parser.parse_args()

    pdf_path = args.pdf_path
    start_page = args.start_page
    forced_amount_unit = args.amount_unit

    if not os.path.exists(pdf_path):
        logging.error(f"Input PDF file not found: {pdf_path}")
        sys.exit(1)

    log_msg = f"Starting PDF parsing for '{pdf_path}' from page {start_page}"
    if forced_amount_unit:
        log_msg += f" (forcing amount_unit='{forced_amount_unit.name}')"
    logging.info(log_msg + "...")

    # Call the main parsing function from the utils module
    all_pages_data = parse_full_pdf_structure(
        pdf_path,
        start_page=start_page,
        forced_amount_unit=forced_amount_unit
    )

    if not all_pages_data:
        logging.warning(f"No data was successfully parsed from '{pdf_path}'. Exiting.")
        sys.exit(0) # Exit gracefully if no data, might not be an error

    logging.info(f"Parsing complete. Validating {len(all_pages_data)} parsed pages...")

    # Validate each page's data
    valid_pages = 0
    for i, page_data in enumerate(all_pages_data):
        # Ensure page_data is a dict before validation
        if isinstance(page_data, dict):
             if validate_page_data(page_data):
                 valid_pages += 1
        else:
            logging.error(f"Encountered non-dictionary item in results at index {i}. Item: {page_data}")
            # Handle this case - maybe remove it or log it? For now, just log.

    logging.info(f"Validation complete. {valid_pages}/{len(all_pages_data)} pages passed basic validation (page_number and title presence).")

    # Determine output path: Same directory and basename as input PDF, but with .json extension
    pdf_dir = os.path.dirname(pdf_path)
    pdf_basename = os.path.basename(pdf_path)
    pdf_name_without_ext = os.path.splitext(pdf_basename)[0]
    output_json_path = os.path.join(pdf_dir, f"{pdf_name_without_ext}.json")

    logging.info(f"Writing {len(all_pages_data)} entries to JSON file: {output_json_path}")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
        logging.info("Successfully wrote JSON output.")
    except IOError as e:
        logging.error(f"Failed to write JSON output to {output_json_path}: {e}")
        sys.exit(1)
    except TypeError as e:
        logging.error(f"Data serialization error writing to JSON: {e}. Check data structure.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 