#!/usr/bin/env python
"""
Script to parse the U.S. Tax Code XML file and insert the data into the database.
Utilizes shared XML parsing utilities.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import date
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project components
from ai_tax_agent.settings import settings
from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session
# Import shared parsing utilities
from ai_tax_agent.parsers.xml_parser_utils import ( 
    find_section_elements, 
    extract_clean_section_number,
    USLM_NS
)

def parse_tax_code(xml_file_path: str):
    """
    Parse the U.S. Tax Code XML file and insert the data into the database.
    
    Args:
        xml_file_path (str): Path to the XML file.
    """
    print(f"Parsing tax code from {xml_file_path}...")
    
    # Parse the XML file
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file_path}")
        return

    # Register namespace globally for ET XPath usage if needed elsewhere, though utils handle it
    ET.register_namespace('', USLM_NS['ns'])
    
    # Get the title number from the XML meta (can also be moved to utils later)
    title_number = None
    meta_element = root.find('.//ns:meta', namespaces=USLM_NS)
    if meta_element is not None:
        doc_number_element = meta_element.find('.//ns:docNumber', namespaces=USLM_NS)
        if doc_number_element is not None and doc_number_element.text:
            try:
                title_number = int(doc_number_element.text)
            except ValueError:
                 print(f"Warning: Could not convert docNumber '{doc_number_element.text}' to integer.")
    if title_number is None:
        print("Warning: Could not find or parse title number (docNumber).")
    
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
    
    # --- Process Sections using Utilities ---
    sections_to_insert = []
    print("Finding and processing section elements...")
    
    # Use the utility function to find section elements
    for section_element in find_section_elements(root, ns=USLM_NS):
        # Use utility function to extract and clean number
        cleaned_number = extract_clean_section_number(section_element, ns=USLM_NS)
        
        if cleaned_number:
            # Extract other details needed for the DB record
            # (These could also be moved to utility functions later)
            heading_element = section_element.find('ns:heading', namespaces=USLM_NS)
            section_title = heading_element.text.strip() if heading_element is not None and heading_element.text else None
            try:
                full_text = ''.join(section_element.itertext()).strip()
            except Exception as e:
                print(f"Warning: Error extracting text from section {cleaned_number}: {e}")
                full_text = None
            special_formatting = False
            if section_element.get('style') and '-uslm-lc:' in section_element.get('style'):
                special_formatting = True
            # Note: Amendment count and hierarchy are not handled by these basic utils yet

            section_data = {
                'title_number': title_number,
                'section_number': cleaned_number,
                'section_title': section_title,
                'full_text': full_text,
                'amendment_count': 0, # Placeholder
                'special_formatting': special_formatting,
                'updated_at': date.today()
                # Add hierarchy fields back as None if needed, or enhance utils
                # 'subtitle': None, 'chapter': None, etc.
            }
            sections_to_insert.append(section_data)
        else:
             # Optionally log sections found but number couldn't be extracted
             pass 
    
    # Insert sections into the database
    section_count = len(sections_to_insert)
    if section_count > 0:
        print(f"Inserting {section_count} sections into the database...")
        # Keep batch insert logic for efficiency if needed, or simple loop for now
        for section_data in tqdm(sections_to_insert, desc="Inserting sections"):
            try:
                section = UsCodeSection(**section_data)
                session.add(section)
            except Exception as e:
                print(f"Error creating UsCodeSection object for data: {section_data}")
                print(f"Error: {e}")
                session.rollback() # Rollback immediately on creation error
                break # Stop insertion on error?
        
        # Commit the changes
        try:
            session.commit()
            print(f"Successfully inserted/updated {section_count} sections.")
        except Exception as e:
            print(f"Error committing sections to the database: {e}")
            session.rollback()
    else:
        print("No valid sections found to insert.")
    
    # Close the session
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the U.S. Tax Code XML file and insert the data into the database.')
    parser.add_argument('--xml-file', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'usc26.xml'),
                        help='Path to the XML file (default: data/usc26.xml)')
    
    args = parser.parse_args()
    parse_tax_code(args.xml_file) 