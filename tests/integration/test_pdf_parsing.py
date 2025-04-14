# tests/integration/test_pdf_parsing.py

import pytest
import json
import os
import sys
import re

# Add project root to sys.path
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from ai_tax_agent.parsers.pdf_parser_utils import parse_pdf_page_structure

# Define paths relative to the test file or workspace root
# Adjust if your test setup is different
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/pdf_parser")
PDF_PATH = "data/tax_statistics/individuals.pdf" # Relative to workspace root

# Helper function to load JSON fixtures
def load_fixture(filename):
    path = os.path.join(FIXTURE_DIR, filename)
    if not os.path.exists(path):
         pytest.fail(f"Fixture file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f: # Ensure utf-8 encoding
        return json.load(f)

def normalize_string(s: str) -> str:
    """Normalizes strings for comparison, handling common variations."""
    if not isinstance(s, str):
        return str(s) # Return string representation if not a string
    # Replace common quote variations with standard ASCII apostrophe
    s = s.replace("\u2019", "'") # Right single quote
    s = s.replace("\u2018", "'") # Left single quote
    s = s.replace("\u201c", '"') # Left double quote
    s = s.replace("\u201d", '"') # Right double quote
    # Optional: Normalize whitespace (replace multiple spaces/newlines with single space)
    # s = ' '.join(s.split())
    return s.strip()

# --- Test Cases --- 

@pytest.mark.parametrize(
    "page_num, expected_fixture_file",
    [
        (26, "page_26_expected.json"),
        (30, "page_30_expected.json"),
    ]
)
def test_pdf_page_structure(page_num, expected_fixture_file):
    """Tests the end-to-end PDF parsing for specific pages against known fixtures."""
    # Construct the absolute path to the PDF if needed, assuming test run from workspace root
    # pdf_abs_path = os.path.join(project_root_path, PDF_PATH)
    # Or keep it relative if the utility function handles it
    pdf_abs_path = PDF_PATH
    
    if not os.path.exists(pdf_abs_path):
         pytest.fail(f"Test PDF file not found: {pdf_abs_path}")

    print(f"\nTesting PDF parsing for page {page_num}...")
    actual_result = parse_pdf_page_structure(pdf_abs_path, page_num)
    
    assert actual_result is not None, f"Parsing failed for page {page_num}"

    expected_result = load_fixture(expected_fixture_file)

    # Compare the results (consider using a deep comparison library if needed)
    # --- Assert Top-Level Structure (Form, Schedule, Unit) ---
    assert actual_result.get("form_title") == expected_result.get("form_title"), "Form title mismatch"
    assert actual_result.get("schedule_title") == expected_result.get("schedule_title"), "Schedule title mismatch"
    assert actual_result.get("amount_unit") == expected_result.get("amount_unit"), "Amount unit mismatch"

    # --- Assert Line Item Number and Label --- 
    # Create dictionaries mapping line number to normalized label
    actual_lines = {str(item['line_item_number']): normalize_string(item.get('label')) 
                    for item in actual_result.get('line_items', []) 
                    if isinstance(item, dict) and 'line_item_number' in item}
    expected_lines = {str(item['line_item_number']): normalize_string(item.get('label')) 
                      for item in expected_result.get('line_items', []) 
                      if isinstance(item, dict) and 'line_item_number' in item}
    
    # Check if the sets of line item numbers are the same
    assert set(actual_lines.keys()) == set(expected_lines.keys()), \
           f"Mismatch in line item numbers found for page {page_num}. Expected: {sorted(expected_lines.keys())}, Actual: {sorted(actual_lines.keys())}"

    # Check if normalized labels match for each common line item number
    mismatched_labels = {}
    for line_num in expected_lines:
        if actual_lines.get(line_num) != expected_lines[line_num]:
            mismatched_labels[line_num] = {
                "expected": expected_lines[line_num],
                "actual": actual_lines.get(line_num)
            }
            
    assert not mismatched_labels, \
        f"Mismatch in normalized line item labels for page {page_num}:\n{json.dumps(mismatched_labels, indent=2)}"

    print(f"Line item numbers and normalized labels match for page {page_num}.")


    # --- Assert Associated Amounts --- 
    actual_amounts = {item['line_item_number']: item.get('amount') for item in actual_result.get('line_items', [])}
    expected_amounts = {item['line_item_number']: item.get('amount') for item in expected_result.get('line_items', [])}

    # Check amounts (this is where the failure is expected)
    assert actual_amounts == expected_amounts, \
           f"Mismatch in associated amounts for page {page_num}. Expected:\n{json.dumps(expected_amounts, indent=2)}\nActual:\n{json.dumps(actual_amounts, indent=2)}"

    print(f"Page {page_num} parsing matched the fixture completely.")

# --- Optional: Add Makefile Target --- 
# Consider adding a target to Makefile:
# test-pdf-parser:
#	poetry run pytest tests/integration/test_pdf_parsing.py -s 