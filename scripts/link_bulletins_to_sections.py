#!/usr/bin/env python
"""
Links IRS Bulletin Items to US Code Sections based on the heuristically
extracted section numbers stored in the referenced_sections field.
"""

import os
import sys
import argparse
from typing import Dict, Set, Optional

from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project components
from ai_tax_agent.settings import settings
from ai_tax_agent.database.models import Base, IrsBulletinItem, UsCodeSection, IrsBulletinItemToCodeSection
from ai_tax_agent.database.session import get_session

def build_section_lookup(session: Session) -> Dict[str, int]:
    """Creates a dictionary mapping section_number (string) to section_id (int)."""
    print("Building US Code Section lookup table...")
    sections = session.query(UsCodeSection.id, UsCodeSection.section_number).all()
    # Handle potential variations in section numbers if needed (e.g., case, spacing)
    # For now, assume section_number in DB matches the cleaned extracted number
    lookup = {str(sec.section_number): sec.id for sec in sections if sec.section_number}
    print(f"Built lookup for {len(lookup)} sections.")
    return lookup

def clean_section_number(num_str: str) -> Optional[str]:
    """Cleans a single extracted section number string.
       Returns None if it contains a decimal (likely a regulation or doc section).
    """
    # Remove leading/trailing whitespace and potentially trailing dots
    cleaned = num_str.strip().rstrip('.')
    
    # Ignore if it contains a decimal point (likely not an IRC section)
    if '.' in cleaned:
        return None
        
    # Optional: Further validation? E.g., ensure it starts with a digit?
    # Add any other specific cleaning rules if needed
    return cleaned

def link_items_to_sections(clear_existing: bool = False, batch_size: int = 1000):
    """Performs the linking process."""
    session = get_session()
    
    if clear_existing:
        print("Clearing existing links in irs_bulletin_item_to_code_section table...")
        try:
            session.query(IrsBulletinItemToCodeSection).delete()
            session.commit()
            print("Existing links cleared.")
        except Exception as e:
            print(f"Error clearing links: {e}")
            session.rollback()
            session.close()
            return

    # 1. Build section lookup
    section_lookup = build_section_lookup(session)
    if not section_lookup:
        print("Error: US Code Section lookup is empty. Cannot perform linking.")
        session.close()
        return

    # 2. Fetch bulletin items with references
    print("Fetching bulletin items with referenced sections...")
    items_to_process = session.query(IrsBulletinItem.id, IrsBulletinItem.referenced_sections)\
                              .filter(IrsBulletinItem.referenced_sections != None)\
                              .all()
    print(f"Found {len(items_to_process)} items with references to process.")

    # 3. Process items and create links
    links_created = 0
    links_skipped_duplicates = 0
    links_failed_lookup = 0
    failed_lookup_examples = []
    max_failed_examples_to_show = 30 # Limit logged examples
    processed_count = 0

    for item_id, ref_section_str in tqdm(items_to_process, desc="Linking Items to Sections"):
        if not ref_section_str:
            continue

        # Parse the comma-delimited string into a set of unique, cleaned numbers
        referenced_numbers: Set[str] = set()
        for num_str in ref_section_str.split(','):
            cleaned_num = clean_section_number(num_str)
            if cleaned_num:
                referenced_numbers.add(cleaned_num)
        
        if not referenced_numbers:
            continue

        # Create links for valid section numbers
        for section_num in referenced_numbers:
            target_section_id = section_lookup.get(section_num)
            
            if target_section_id:
                # Create the association object
                link = IrsBulletinItemToCodeSection(
                    bulletin_item_id=item_id,
                    section_id=target_section_id
                    # relevance_notes is removed
                )
                session.add(link)
                # Try flushing within the loop to catch integrity errors early
                try:
                    session.flush() 
                    links_created += 1
                except IntegrityError: # Handles unique constraint violation
                    session.rollback() # Important: Rollback the specific failed insertion
                    links_skipped_duplicates += 1
                    # print(f"Skipped duplicate link: item {item_id} -> section {target_section_id} ({section_num})")
                except Exception as e:
                    session.rollback()
                    print(f"\nError adding link for item {item_id} -> section {target_section_id} ({section_num}): {e}")
            else:
                links_failed_lookup += 1
                # Log unmatched section numbers (up to a limit)
                if len(failed_lookup_examples) < max_failed_examples_to_show:
                    failed_lookup_examples.append(section_num)
                # print(f"Warning: Referenced section number '{section_num}' from item {item_id} not found in us_code_section lookup.") # Old print
        
        processed_count += 1
        # Commit periodically
        if processed_count % batch_size == 0:
            try:
                # print(f"Committing batch of {batch_size} items' links...")
                session.commit()
            except Exception as e:
                print(f"Error committing batch: {e}")
                session.rollback()
                # Decide if you want to stop or continue after a batch commit error

    # Commit any remaining changes
    try:
        print("Committing final links...")
        session.commit()
    except Exception as e:
        print(f"Error committing final links: {e}")
        session.rollback()

    print("\n--- Linking Summary ---")
    print(f"Links created: {links_created}")
    print(f"Links skipped (duplicates): {links_skipped_duplicates}")
    print(f"Links failed (section not found): {links_failed_lookup}")
    if failed_lookup_examples:
        print("\n--- Examples of Failed Section Lookups (Cleaned Extracted Number) ---")
        print(", ".join(failed_lookup_examples))
        print("(Check if these numbers/formats exist in the us_code_section table)")
        print("----------------------------------------------------------------")

    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link IRS Bulletin Items to US Code Sections.")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing links before processing.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of items to process before committing links.")
    
    args = parser.parse_args()
    link_items_to_sections(clear_existing=args.clear, batch_size=args.batch_size) 