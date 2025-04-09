"""
Utilities for parsing USLM XML files, specifically Title 26 (U.S. Tax Code).
"""

import xml.etree.ElementTree as ET
from typing import Iterator, Optional, Dict, Any

# Define the standard USLM namespace
USLM_NS = {'ns': 'http://xml.house.gov/schemas/uslm/1.0'}

def find_section_elements(xml_root: ET.Element, ns: Dict[str, str] = USLM_NS) -> Iterator[ET.Element]:
    """Finds and yields all relevant <section> elements within the main <title>.

    Args:
        xml_root: The root element of the parsed XML tree.
        ns: The namespace dictionary to use for searching.

    Yields:
        Relevant <section> ET.Element objects.
    """
    main_element = xml_root.find('.//ns:main', namespaces=ns)
    if main_element is None:
        print("Warning: Could not find main element (ns:main) in XML.")
        return
        
    title_element = main_element.find('.//ns:title', namespaces=ns)
    if title_element is None:
        print("Warning: Could not find title element (ns:title) within main element.")
        return
        
    # Use findall to get direct children and potentially grandchildren if structure varies
    # Using .// ensures we find sections even if nested deeper than expected
    section_elements = title_element.findall('.//ns:section', namespaces=ns)
    
    for section_element in section_elements:
        # Skip inline sections often used for notes within text
        if section_element.get('class') == 'inline':
            continue
        yield section_element

def extract_clean_section_number(section_element: ET.Element, ns: Dict[str, str] = USLM_NS) -> Optional[str]:
    """Extracts and cleans the section number from a <section> element.
    
    Args:
        section_element: The <section> element.
        ns: The namespace dictionary.

    Returns:
        The cleaned section number string, or None if not found/extracted.
    """
    num_element = section_element.find('ns:num', namespaces=ns) # Use relative path from section
    if num_element is None:
        # --- Debugging --- 
        element_id = section_element.get('identifier', 'No identifier')
        print(f"[Debug] Found <section> ({element_id}) but could not find 'ns:num' sub-element.")
        # --------------- 
        return None
    
    section_number_raw = num_element.get('value', '')
    if not section_number_raw and num_element.text:
        section_number_raw = num_element.text.strip()
    
    if not section_number_raw:
        # --- Debugging --- 
        element_id = section_element.get('identifier', 'No identifier')
        num_text = num_element.text if num_element.text else 'None'
        print(f"[Debug] Found 'ns:num' in <section> ({element_id}), but 'value' attribute and text content ('{num_text}') were empty or missing.")
        # --------------- 
        return None

    # Cleaning logic (matching parse_tax_code.py and test)
    section_number = section_number_raw.replace('§', '').strip()
    if section_number.endswith('.'):
        section_number = section_number[:-1]
    cleaned_number = section_number.strip()
    
    # Return None if cleaning results in an empty string
    if not cleaned_number:
         # --- Debugging --- 
        element_id = section_element.get('identifier', 'No identifier')
        print(f"[Debug] Raw section number ('{section_number_raw}') in <section> ({element_id}) resulted in empty string after cleaning.")
        # --------------- 
        return None
        
    return cleaned_number

# Optional: Add more extraction functions here later as needed
# e.g., extract_section_title, extract_full_text, extract_hierarchy etc. 