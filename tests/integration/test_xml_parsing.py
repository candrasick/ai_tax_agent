# tests/integration/test_xml_parsing.py

import os
import sys
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path

# Ensure the main project directory is in the path for imports 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import shared parsing utilities
from ai_tax_agent.parsers.xml_parser_utils import (
    find_section_elements,
    extract_clean_section_number,
    USLM_NS # Use the namespace from the utils
)

# --- Constants ---
# Assuming the script runs from the project root or pytest handles paths correctly
XML_FILE_PATH = Path("data/usc26.xml")
# Define the XML namespace (same as in parse_tax_code.py)
NS = {'ns': 'http://xml.house.gov/schemas/uslm/1.0'}
TARGET_SECTION = "11" # The section we want to ensure is found (changed from 1108 which doesn't exist)

# --- Helper: Extraction/Cleaning Logic (adapted from parse_tax_code.py) ---

def _extract_and_clean_number(section_element, ns):
    """Extracts and cleans the section number from a <section> element."""
    num_element = section_element.find('.//ns:num', namespaces=ns)
    if num_element is None:
        return None
    
    section_number_raw = num_element.get('value', '')
    if not section_number_raw and num_element.text:
        section_number_raw = num_element.text.strip()
    
    if not section_number_raw:
        return None

    # Apply the same cleaning logic as used in parse_tax_code.py
    section_number = section_number_raw.replace('§', '').strip()
    if section_number.endswith('.'):
        section_number = section_number[:-1]
    cleaned_number = section_number.strip()
    
    # Return None if cleaning results in an empty string
    return cleaned_number if cleaned_number else None

# --- The Test Function (Now using utils) ---

def test_xml_section_parsing():
    """Parses usc26.xml in memory using shared utils, checks for section 1108 and duplicates."""
    
    if not XML_FILE_PATH.is_file():
        pytest.fail(f"XML file not found at expected path: {XML_FILE_PATH.resolve()}")

    print(f"\nParsing XML file: {XML_FILE_PATH}")
    extracted_numbers = []
    
    try:
        tree = ET.parse(XML_FILE_PATH)
        root = tree.getroot()
    except ET.ParseError as e:
        pytest.fail(f"Error parsing XML file: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error opening/reading XML file: {e}")

    # Register namespace (optional, as utils pass it explicitly)
    # ET.register_namespace('', USLM_NS['ns'])
        
    # Iterate through sections using the utility function
    found_elements_count = 0
    for section_element in find_section_elements(root, ns=USLM_NS):
        found_elements_count += 1
        # Extract number using the utility function
        cleaned_number = extract_clean_section_number(section_element, ns=USLM_NS)
        if cleaned_number:
            extracted_numbers.append(cleaned_number)
            
    print(f"Found {found_elements_count} potential <section> elements using utility function.")
    print(f"Extracted {len(extracted_numbers)} cleaned section numbers using utility function.")

    # --- Assertions ---
    assert extracted_numbers, "No section numbers were extracted by the utility functions."
    
    # 1. Check if the target section is found
    assert TARGET_SECTION in extracted_numbers, \
        f"Target section '{TARGET_SECTION}' was not found in the extracted section numbers."
    
    # 2. Check for duplicates
    duplicates = {num for num in extracted_numbers if extracted_numbers.count(num) > 1}
    assert len(duplicates) == 0, f"Duplicate section numbers found: {duplicates}"
    print(f"Assertion PASSED: No duplicate section numbers found.") 