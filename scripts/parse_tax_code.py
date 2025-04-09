#!/usr/bin/env python
"""
Script to parse the U.S. Tax Code XML file and insert the data into the database.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import date
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session

# Define the XML namespace
NS = {'ns': 'http://xml.house.gov/schemas/uslm/1.0'}

# Global list to store section data before insertion
SECTIONS_TO_INSERT = []

def parse_tax_code(xml_file_path):
    """
    Parse the U.S. Tax Code XML file and insert the data into the database.
    
    Args:
        xml_file_path (str): Path to the XML file.
    """
    global SECTIONS_TO_INSERT
    SECTIONS_TO_INSERT = [] # Reset global list

    print(f"Parsing tax code from {xml_file_path}...")
    
    # Parse the XML file
    try:
        # Use iterparse for potentially large files, though full parsing is needed for context
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file_path}")
        return

    # Register namespace globally for ET XPath usage
    ET.register_namespace('', NS['ns'])
    
    # Get the title number from the XML
    title_number = None
    meta_element = root.find('.//ns:meta', namespaces=NS)
    if meta_element is not None:
        doc_number_element = meta_element.find('.//ns:docNumber', namespaces=NS)
        if doc_number_element is not None and doc_number_element.text:
            try:
                title_number = int(doc_number_element.text)
            except ValueError:
                 print(f"Warning: Could not convert docNumber '{doc_number_element.text}' to integer.")
    
    if title_number is None:
        print("Warning: Could not find or parse title number (docNumber) in XML meta element.")
    
    # Create a database session
    session = get_session()
    
    # --- Delete existing data ---
    try:
        print("Deleting existing data from us_code_section table...")
        num_deleted = session.query(UsCodeSection).delete()
        session.commit()
        print(f"Deleted {num_deleted} existing rows.")
    except Exception as e:
        print(f"Error deleting existing data: {e}")
        session.rollback()
        session.close()
        return
    # --------------------------

    # Process the main content
    main_element = root.find('.//ns:main', namespaces=NS)
    if main_element is None:
        print("Error: Could not find main element (ns:main) in XML.")
        session.close()
        return
    
    # Process the title element within main
    title_element = main_element.find('.//ns:title', namespaces=NS)
    if title_element is None:
        print("Error: Could not find title element (ns:title) within main element in XML.")
        session.close()
        return
    
    # Start recursive processing from the children of the title element
    initial_context = {
        'subtitle': None,
        'chapter': None,
        'subchapter': None,
        'part': None,
        'subpart': None
    }
    print("Starting recursive processing of XML structure...")
    # Wrap the iteration with tqdm if possible, though might be complex with recursion
    # total_elements = sum(1 for _ in title_element.iter())
    # with tqdm(total=total_elements, desc="Processing XML Elements") as pbar:
    for child in title_element:
        process_element(child, initial_context, title_number)
            # pbar.update(1) # Manual update needed if tqdm is used here

    # Insert collected sections into the database
    section_count = len(SECTIONS_TO_INSERT)
    if section_count > 0:
        print(f"Inserting {section_count} sections into the database...")
        for section_data in tqdm(SECTIONS_TO_INSERT, desc="Inserting sections"):
            try:
                section = UsCodeSection(**section_data)
                session.add(section)
            except Exception as e:
                print(f"Error creating UsCodeSection object for data: {section_data}")
                print(f"Error: {e}")
                # Optionally rollback or skip this section
        
        # Commit the changes
        try:
            session.commit()
            print(f"Successfully inserted {section_count} sections into the database.")
        except Exception as e:
            print(f"Error committing sections to the database: {e}")
            session.rollback()
    else:
        print("No valid sections found to insert.")
    
    # Close the session
    session.close()

def get_element_heading(element):
    """Helper to get heading text from common hierarchy elements."""
    heading_el = element.find('ns:heading', namespaces=NS)
    if heading_el is not None and heading_el.text:
        return heading_el.text.strip()
    # Sometimes the identifier might be in the num tag
    num_el = element.find('ns:num', namespaces=NS)
    if num_el is not None and num_el.text:
         return num_el.text.strip()
    return None

def process_element(element, context, title_number):
    """
    Recursively process XML elements, updating hierarchy context and extracting sections.
    """
    global SECTIONS_TO_INSERT
    
    tag_name = element.tag.replace(f"{{{NS['ns']}}}", "") # Get tag name without namespace
    
    # Create a new context for children of this element
    new_context = context.copy()
    
    # Update context based on current element type
    if tag_name == 'subtitle':
        new_context['subtitle'] = get_element_heading(element)
        # Reset lower levels when a higher level changes
        new_context['chapter'] = None
        new_context['subchapter'] = None
        new_context['part'] = None
        new_context['subpart'] = None
    elif tag_name == 'chapter':
        new_context['chapter'] = get_element_heading(element)
        new_context['subchapter'] = None
        new_context['part'] = None
        new_context['subpart'] = None
    elif tag_name == 'subchapter':
        new_context['subchapter'] = get_element_heading(element)
        new_context['part'] = None
        new_context['subpart'] = None
    elif tag_name == 'part':
        new_context['part'] = get_element_heading(element)
        new_context['subpart'] = None
    elif tag_name == 'subpart':
        new_context['subpart'] = get_element_heading(element)
        
    elif tag_name == 'section':
        # Skip inline sections
        if element.get('class') != 'inline':
            section_data = extract_section_data(element, title_number, context) # Pass the *current* context
            if section_data:
                SECTIONS_TO_INSERT.append(section_data)
        # Don't recurse further into sections for hierarchy context
        # Sections contain their own text/subsections not relevant for outer hierarchy
        return

    # Recursively process children
    for child in element:
        process_element(child, new_context, title_number)
        # if pbar: pbar.update(1) # Manual update if tqdm is used


def extract_section_data(section_element, title_number, hierarchy_context):
    """
    Extract section data from a section element, including hierarchy context.
    
    Args:
        section_element (Element): The section element.
        title_number (int): The title number.
        hierarchy_context (dict): The hierarchy context (subtitle, chapter, etc.).
        
    Returns:
        dict or None: A dictionary containing the section data, or None if essential data is missing.
    """
    # Extract section number
    num_element = section_element.find('.//ns:num', namespaces=NS)
    if num_element is None:
        return None
    
    section_number_raw = num_element.get('value', '')
    if not section_number_raw and num_element.text:
        section_number_raw = num_element.text.strip()
    
    if not section_number_raw:
        return None

    # Clean the section number (Example: "§ 1." -> "1", "12A" -> "12A")
    section_number = section_number_raw.replace('§', '').strip()
    if section_number.endswith('.'):
        section_number = section_number[:-1]
    section_number = section_number.strip()
    
    # Extract section title
    heading_element = section_element.find('.//ns:heading', namespaces=NS)
    section_title = heading_element.text.strip() if heading_element is not None and heading_element.text else None
    
    # Extract full text
    try:
        full_text = ''.join(section_element.itertext()).strip()
    except Exception as e:
        print(f"Warning: Error extracting text from section {section_number_raw}: {e}")
        full_text = None
    
    # Check if the section has special formatting
    special_formatting = False
    if section_element.get('style') and '-uslm-lc:' in section_element.get('style'):
        special_formatting = True
    
    # Count amendments
    amendment_count = count_amendments(section_element)
    
    # Create section data dictionary, including hierarchy from context
    section_data = {
        'title_number': title_number,
        'subtitle': hierarchy_context.get('subtitle'),
        'chapter': hierarchy_context.get('chapter'),
        'subchapter': hierarchy_context.get('subchapter'),
        'part': hierarchy_context.get('part'),
        'subpart': hierarchy_context.get('subpart'),
        'section_number': section_number,
        'section_title': section_title,
        'full_text': full_text,
        'amendment_count': amendment_count,
        'special_formatting': special_formatting,
        'updated_at': date.today()
    }
    
    return section_data

def count_amendments(section_element):
    """
    Count the number of amendments noted within a section element.
    
    Args:
        section_element (Element): The section element.
        
    Returns:
        int: The estimated number of amendments based on specific tags/text.
    """
    amendment_count = 0
    
    # Count specific <amendment> elements if they exist (adjust tag name if needed)
    amendment_elements = section_element.findall('.//ns:amendment', namespaces=NS)
    amendment_count += len(amendment_elements)
    
    # Count <note> elements that seem related to amendments
    notes_section = section_element.find('.//ns:notes', namespaces=NS)
    if notes_section is not None:
        for note in notes_section.findall('.//ns:note', namespaces=NS):
            # Check if the note itself is categorized as 'Amendments'
            if note.get('topic') == 'amendments' or note.get('type') == 'Amendments':
                 amendment_count += 1 # Count the note itself as one amendment block
                 continue # Avoid double counting text within this note

            note_text = ''.join(note.itertext()).lower()
            # Simple check for keywords within other notes
            if 'amended by pub. l.' in note_text or 'amendments' in note_text:
                 # This might overcount if one note details multiple amendments
                 # Consider more sophisticated parsing if needed
                 amendment_count += 1 
            
    return amendment_count

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Parse the U.S. Tax Code XML file and insert the data into the database.')
    parser.add_argument('--xml-file', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'usc26.xml'),
                        help='Path to the XML file (default: data/usc26.xml)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Parse the tax code
    parse_tax_code(args.xml_file) 