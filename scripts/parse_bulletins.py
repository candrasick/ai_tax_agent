#!/usr/bin/env python
"""
Heuristic parser for IRS Bulletin PDFs to populate database tables.
Uses regex based heuristics to find items, actions, and section references.
"""

import os
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from tqdm import tqdm
from sqlalchemy.orm import Session
import pypdf # Using pypdf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project components
from ai_tax_agent.settings import settings
from ai_tax_agent.database.models import Base, IrsBulletin, IrsBulletinItem
from ai_tax_agent.database.session import get_session, engine

# --- Configuration & Constants ---
BULLETIN_DIR = "data/bulletins"
# Regex to extract year and number from filename like irb24-08.pdf
FILENAME_REGEX = re.compile(r"irb(\d{2})-(\d+)\.pdf", re.IGNORECASE)

# Regex to find potential item start headers (Based on User Input)
# Groups: type, number, title, page
ITEM_HEADER_REGEX = re.compile(
    r"^\s*(?P<type>T\.D\.|Notice|Rev\.\s*Proc\.|Rev\.\s*Rul\.)\s*"  # Item Type (Start of line)
    r"(?P<number>\d{4}[–-]\d+|\d+-\d+|\d+)\s*"                # Item Number (YYYY-N, N-N, or N)
    r"(?P<title>.*?)\s*"                                      # Title (Non-greedy match until page or end)
    r"(?:page\s*(?P<page>\d+))?$\s*",                         # Optional Page number at end of line
    re.IGNORECASE | re.MULTILINE
)

# Regex for common action keywords
ACTION_KEYWORDS = ['amplified', 'clarified', 'corrected', 'modified', 'obsoleted',
                 'revoked', 'superseded', 'supplemented', 'suspended']
ACTION_REGEX = re.compile(r'\b(' + '|'.join(ACTION_KEYWORDS) + r')\b', re.IGNORECASE)

# Regex for potential code section references (Refined)
# Allows for more flexible spacing and prefixes
SECTION_REF_REGEX = re.compile(
    r"(?:section|sec\.|§{1,2}|irc)\s+"  # Prefixes (section, sec., §, §§, IRC) followed by space/newline
    r"(\d+[\w.-]*)"                     # Group 1: Digits followed by word chars, dot, or hyphen
    r"(?:\s*\([a-z0-9]+\))?"           # Optional SINGLE parenthesized subsection like (a) or (1)
    , re.IGNORECASE
)

# --- Helper Functions ---

def parse_bulletin_filename(filename: str) -> Optional[Tuple[str, str, Optional[datetime.date]]]:
    """Parses year and number from filename, estimates date."""
    match = FILENAME_REGEX.match(filename)
    if match:
        year_short_str, number_str = match.groups()
        try:
            year_short = int(year_short_str)
            if year_short >= 95:
                year = 1900 + year_short
            else:
                year = 2000 + year_short
            week_num = int(number_str)
            estimated_date = datetime.strptime(f'{year}-W{week_num:02d}-1', "%G-W%V-%u")
        except ValueError:
            print(f"Warning: Could not parse date/year info from {filename}")
            return None, None, None
        return str(year), number_str, estimated_date.date()
    return None, None, None

def extract_text_from_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """Extracts text from the first few pages of a PDF."""
    text = ""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        num_pages_to_read = len(reader.pages)
        if max_pages:
            num_pages_to_read = min(num_pages_to_read, max_pages)
        
        for i in range(num_pages_to_read):
            try:
                page = reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n" # Add page break marker
            except Exception as page_e:
                 print(f"\nWarning: Error extracting text from page {i+1} of {pdf_path.name}: {page_e}")
                 continue # Try next page
                 
    except pypdf.errors.PdfReadError as pdf_err:
         print(f"\nError reading PDF {pdf_path.name} (pypdf error): {pdf_err}")
    except Exception as e:
        # Catch other potential errors like file access issues
        print(f"\nUnexpected error reading PDF {pdf_path.name}: {e}")
    return text

def find_item_segments(full_text: str) -> List[Dict]:
    """Attempts to segment the text based on revised item headers."""
    segments = []
    matches = list(ITEM_HEADER_REGEX.finditer(full_text))
    
    for i, match in enumerate(matches):
        start_index = match.start()
        end_index = matches[i+1].start() if (i + 1) < len(matches) else len(full_text)
        
        segment_text = full_text[start_index:end_index].strip()
        match_dict = match.groupdict()
        item_type = match_dict.get('type', 'Other').strip().replace('.', '')
        item_number = match_dict.get('number', 'Unknown').strip()
        # Clean up title: remove trailing dots/spaces, check if empty
        item_title = match_dict.get('title')
        if item_title:
             item_title = item_title.strip()
             # Remove trailing dots potentially captured before page number
             item_title = re.sub(r'\.+\s*$', '', item_title).strip()
        item_title = item_title if item_title else None # Set to None if empty after cleaning

        page_start_str = match_dict.get('page')
        try:
            page_start = int(page_start_str) if page_start_str else None
        except ValueError:
            page_start = None

        segments.append({
            "type": item_type,
            "number": item_number,
            "title": item_title, # Add title
            "page": page_start, 
            "text": segment_text,
            "start_index": start_index,
        })
    return segments

def extract_details_from_segment(segment_text: str) -> Dict:
    """Extracts actions and specific section numbers from a text segment."""
    details = {"actions": set(), "section_numbers": set()}
    
    # Find actions
    for match in ACTION_REGEX.finditer(segment_text):
        details["actions"].add(match.group(1).lower())
        
    # --- DEBUG: Print segment text before searching for sections ---
    # print("\n--- Searching for sections in segment: ---")
    # print(segment_text[:500] + ("..." if len(segment_text) > 500 else "")) # Print start of segment
    # print("-----------------------------------------")
    # ----------------------------------------------------------
        
    # Find section reference numbers (using Group 1 from the refined regex)
    found_any_match = False
    for match in SECTION_REF_REGEX.finditer(segment_text):
        found_any_match = True
        section_num_part = match.group(1) # Get the core number part
        if section_num_part:
            section_num_stripped = section_num_part.strip()
            details["section_numbers"].add(section_num_stripped)
            # --- DEBUG: Print found section number --- 
            # print(f"  DEBUG: Found section ref -> {match.group(0)} -> Captured: {section_num_stripped}")
            # ----------------------------------------
    
    # --- DEBUG: Indicate if no matches were found ---
    # if not found_any_match:
    #     print("  DEBUG: No section references found by regex in this segment.")
    # ------------------------------------------------
        
    return details

# --- Main Parsing Logic (Updated) ---

def parse_and_load_bulletins(bulletin_dir: str, clear_existing: bool = False, limit: Optional[int] = None):
    """Finds bulletin PDFs, parses them using heuristics, and loads data."""
    pdf_dir = Path(bulletin_dir)
    if not pdf_dir.is_dir():
        print(f"Error: Bulletin directory not found: {bulletin_dir}")
        return

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    if limit:
        pdf_files = pdf_files[:limit]
        print(f"Processing a limit of {limit} PDF files.")
        
    if not pdf_files:
        print(f"No PDF files found or selected in {bulletin_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    session = get_session()

    if clear_existing:
        print("Clearing existing IRS Bulletin data...")
        try:
            session.query(IrsBulletinItem).delete() 
            session.query(IrsBulletin).delete()     
            session.commit()
            print("Existing data cleared.")
        except Exception as e:
            print(f"Error clearing data: {e}")
            session.rollback()
            session.close()
            return
            
    processed_files = 0
    failed_files = 0
    total_items_created = 0

    for pdf_path in tqdm(pdf_files, desc="Parsing Bulletins"):
        filename = pdf_path.name
        year_str, num_str, bulletin_date = parse_bulletin_filename(filename)

        if not year_str or not num_str:
            print(f"\nSkipping {filename}: Could not parse year/number.")
            failed_files += 1
            continue
        
        bulletin_number = f"{year_str}-{int(num_str):02d}"
        source_url = f"https://www.irs.gov/pub/irs-irbs/{filename}"

        # --- Create or Get IrsBulletin --- 
        try:
            bulletin = session.query(IrsBulletin).filter_by(bulletin_number=bulletin_number).first()
            if not bulletin:
                bulletin = IrsBulletin(
                    bulletin_number=bulletin_number,
                    bulletin_date=bulletin_date, 
                    title=f"Internal Revenue Bulletin {bulletin_number}",
                    source_url=source_url
                )
                session.add(bulletin)
                session.flush() # Get ID
            
            # --- Extract Full Text and Segment --- 
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text:
                print(f"\nWarning: Could not extract text from {filename}. Skipping item parsing.")
                failed_files += 1
                session.rollback()
                continue 

            item_segments = find_item_segments(full_text)
            items_created_in_bulletin = 0

            if not item_segments:
                 print(f"\nWarning: No item headers found matching regex in {filename}. Only bulletin record created/updated.")
            else:
                for segment in item_segments:
                    item_type = segment["type"]
                    item_number = segment["number"]
                    item_title = segment["title"] # Get title from segment dict
                    segment_text = segment["text"]
                    
                    details = extract_details_from_segment(segment_text)
                    actions = ", ".join(sorted(list(details["actions"]))) or None
                    section_numbers_found = sorted(list(details["section_numbers"]))
                    referenced_sections_str = ", ".join(section_numbers_found) if section_numbers_found else None

                    existing_item = session.query(IrsBulletinItem).filter_by(
                        bulletin_id=bulletin.id, 
                        item_number=item_number,
                        item_type=item_type
                    ).first()

                    if not existing_item:
                        item = IrsBulletinItem(
                            bulletin_id=bulletin.id,
                            item_type=item_type,
                            item_number=item_number,
                            title=item_title, # Populate title
                            action=actions,
                            full_text=segment_text, 
                            referenced_sections=referenced_sections_str
                        )
                        session.add(item)
                        items_created_in_bulletin += 1

            total_items_created += items_created_in_bulletin
            processed_files += 1
            session.commit()

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            session.rollback()
            failed_files += 1

    print("\n--- Parsing Summary ---")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed processing:    {failed_files} files")
    print(f"Total items created: {total_items_created}")
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse IRS Bulletin PDFs using heuristics and load data.")
    parser.add_argument("--dir", type=str, default=BULLETIN_DIR,
                        help=f"Directory containing bulletin PDF files (default: {BULLETIN_DIR})")
    parser.add_argument("--clear", action="store_true", 
                        help="Clear existing bulletin data from tables before parsing.")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit the number of PDFs to process (for testing). Default: process all.")
    
    args = parser.parse_args()
    parse_and_load_bulletins(bulletin_dir=args.dir, clear_existing=args.clear, limit=args.limit) 