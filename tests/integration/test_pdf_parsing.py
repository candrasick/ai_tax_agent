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
    with open(path, 'r') as f:
        return json.load(f)

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
    actual_lines = {item['line_item_number']: item['label'] for item in actual_result.get('line_items', [])}
    expected_lines = {item['line_item_number']: item['label'] for item in expected_result.get('line_items', [])}
    
    # Check if the sets of line item numbers are the same
    assert set(actual_lines.keys()) == set(expected_lines.keys()), \
           f"Mismatch in line item numbers found for page {page_num}. Expected: {sorted(expected_lines.keys())}, Actual: {sorted(actual_lines.keys())}"

    # Check if the labels match for each line item number
    # Sort items by line item number before comparing for consistent diffs
    sorted_actual_lines = sorted(actual_result.get('line_items', []), key=lambda x: (int(re.match(r"(\d+)", x['line_item_number']).group(1)), x['line_item_number']))
    sorted_expected_lines = sorted(expected_result.get('line_items', []), key=lambda x: (int(re.match(r"(\d+)", x['line_item_number']).group(1)), x['line_item_number']))
    actual_lines_for_diff = [{k: v for k, v in item.items() if k != 'amount'} for item in sorted_actual_lines]
    expected_lines_for_diff = [{k: v for k, v in item.items() if k != 'amount'} for item in sorted_expected_lines]

    assert actual_lines_for_diff == expected_lines_for_diff, \
           f"Mismatch in line item/labels for page {page_num}. Expected:\n{json.dumps(expected_lines_for_diff, indent=2)}\nActual:\n{json.dumps(actual_lines_for_diff, indent=2)}"
    print(f"Line item numbers and labels match for page {page_num}.")


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