#!/usr/bin/env python
"""
Notebook-friendly script to fetch data for specific U.S. Code sections
from the database and save it as a JSON file.
"""

import os
import sys
import json
from typing import List, Dict, Any

from sqlalchemy.orm import Session

# Add the project root to the Python path (adjust based on script location)
# Assuming this script is in scripts/notebook/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Project components
# No LLM imports needed here
from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session

# --- Configuration ---
# Define specific sections to fetch
SECTIONS_TO_FETCH = ['1', '61', '7701'] # Same demo sections
# Define output file path
OUTPUT_FILENAME = os.path.join(os.path.dirname(__file__), "demo_sections_data.json")

# --- Database Function ---
def get_demo_sections_data(session: Session, section_numbers: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches specific sections and returns their data as a list of dictionaries.
    """
    print(f"Fetching data for sections {section_numbers} from database...")
    sections_data = []
    try:
        sections = session.query(UsCodeSection).filter(UsCodeSection.section_number.in_(section_numbers)).all()
        print(f"Found {len(sections)} matching sections.")

        # Sort sections according to the original list order if needed
        sections.sort(key=lambda s: section_numbers.index(s.section_number) if s.section_number in section_numbers else float('inf'))

        # Convert ORM objects to dictionaries for JSON serialization
        for section in sections:
            sections_data.append({
                "id": section.id,
                "section_number": section.section_number,
                "section_title": section.section_title,
                "core_text": section.core_text,
                "amendments_text": section.amendments_text,
                "full_text": section.full_text, # Include full text for reference
                "updated_at": section.updated_at.isoformat() if section.updated_at else None,
                "title_number": section.title_number,
                "subtitle": section.subtitle,
                "chapter": section.chapter,
                "subchapter": section.subchapter,
                "part": section.part,
                "subpart": section.subpart,
                "special_formatting": section.special_formatting
            })
        return sections_data
    except Exception as e:
        print(f"Error fetching demo sections: {e}")
        return []

# --- Main Logic ---
def fetch_and_save_data():
    """Fetches data for demo sections and saves it to JSON."""
    print(f"Starting data fetch for sections: {SECTIONS_TO_FETCH}")

    session = get_session()
    fetched_data = []

    try:
        fetched_data = get_demo_sections_data(session, SECTIONS_TO_FETCH)

        if not fetched_data:
            print("Did not find the specified demo sections in the database.")
            return

        # Save the data to JSON
        print(f"Attempting to save data for {len(fetched_data)} sections to {OUTPUT_FILENAME}...")
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                # Use indent for readability
                json.dump(fetched_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved data to {OUTPUT_FILENAME}")
        except IOError as e:
            print(f"Error writing JSON file: {e}")
        except TypeError as e:
            print(f"Error serializing data to JSON (check data types): {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # import traceback
        # print(traceback.format_exc())
    finally:
        if session:
            session.close()
            print("Database session closed.")

if __name__ == "__main__":
    fetch_and_save_data() 