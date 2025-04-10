#!/usr/bin/env python
"""
Spider to scrape IRS form instructions from https://www.irs.gov/instructions
Extracts form numbers and their corresponding links and stores them in a database.

Options:
  --clear    Clear existing form instructions before scraping new ones
"""
import re
import os
import requests
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from sqlalchemy.exc import IntegrityError

# Import models and session from the project structure
from ai_tax_agent.database.models import FormInstruction, FormField
from ai_tax_agent.database.session import get_session

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('irs_instruction_scraper')

# Base URL for IRS form instructions (adjust if needed)
BASE_URL = "https://www.irs.gov"
FORMS_URL = "https://www.irs.gov/forms-pubs/about-form-{form_number}" # Example, might need adjustment

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Scrape IRS form instructions.")
    parser.add_argument('--clear', action='store_true', 
                        help='Clear existing form instructions before scraping new ones')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompts')
    return parser.parse_args()

def scrape_irs_instructions(session, clear=False, force=False):
    """Scrape IRS form instructions and save to database."""
    logger.info("Starting IRS form instruction scraping...")
    
    if clear:
        count = session.query(FormInstruction).count()
        if count > 0:
            if not force:
                should_continue = input(f"This will delete {count} existing form instructions. Continue? (y/n): ").lower()
                if should_continue != 'y':
                    logger.info("Operation cancelled.")
                    return
            logger.info(f"Clearing {count} existing form instructions...")
            session.query(FormInstruction).delete()
            session.commit()
        else:
            logger.info("No existing form instructions to clear.")

    # --- Start Refined Scraping Logic --- #
    try:
        directory_url = "https://www.irs.gov/instructions"
        logger.info(f"Fetching form directory: {directory_url}")
        response = requests.get(directory_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Selector for the main table body containing the form links
        # Based on the provided HTML structure with class="pup-table"
        table_body_selector = 'table.pup-table tbody'
        table_body = soup.select_one(table_body_selector)

        if not table_body:
            logger.error(f"Could not find the main table body using selector '{table_body_selector}'. Please verify the selector against the page structure.")
            return

        # Find all rows within the table body
        rows = table_body.find_all('tr')
        logger.info(f"Found {len(rows)} rows in the table.")

        form_data_to_process = []
        for row in rows:
            # Find all table data cells (`td`) in the current row
            cells = row.find_all('td')
            # Check if the row has at least 2 cells (Title, HTML Instruction)
            if len(cells) >= 2:
                # Target the second cell (index 1), which contains the HTML link
                html_link_cell = cells[1]
                link_tag = html_link_cell.find('a') # Find the <a> tag within this cell

                if link_tag:
                    original_title = link_tag.get_text(strip=True)
                    href = link_tag.get('href')

                    # Ensure we got a valid link and title, and it's not a PDF link (double check)
                    if href and original_title and not href.lower().endswith('.pdf'):
                        # Ensure URL is absolute
                        full_url = href if href.startswith('http') else BASE_URL + href
                        form_data_to_process.append((original_title, full_url))
                        logger.debug(f"Adding HTML instruction: '{original_title}' - {full_url}")
                    else:
                        logger.debug(f"Skipping link in second cell (not valid HTML link or missing data): {link_tag.prettify()}")
                else:
                    logger.debug("No <a> tag found in the second cell of this row.")
            else:
                logger.debug("Row does not have the expected number of cells (needs at least 2). Skipping row.")

        if not form_data_to_process:
            logger.warning("No valid HTML instruction links extracted from the table rows. Check selectors and table structure.")
            return

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch or parse the main forms directory: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during initial scraping: {e}")
        return
    # --- End Refined Scraping Logic --- #

    saved_count = 0
    updated_count = 0
    skipped_count = 0

    logger.info(f"Processing {len(form_data_to_process)} found HTML instruction items...")

    for original_title, link_url in tqdm(form_data_to_process, desc="Processing Forms"):

        # --- Start REVISED Parsing Logic V3 --- #
        title = original_title # Use the original link text as the title
        form_numbers_found = []

        # 1. Check for non-English prefixes first and skip silently
        non_english_prefixes = ['Instrucciones', '表格'] # Add other prefixes if needed
        is_non_english = any(original_title.startswith(prefix) for prefix in non_english_prefixes)
        if is_non_english:
            logger.debug(f"Skipping non-English title: '{original_title}'")
            skipped_count += 1
            continue

        # 2. Define patterns that capture the CORE form number/schedule identifier
        #    These patterns avoid capturing the date/year in parentheses
        patterns_to_find = [
            r'(?:Form|Schedule)\s+([\w\d-]+(?:\([A-Z]+\))?)', # Form/Schedule XXX, Form/Schedule XXX(PR)
            r'(W-\d+[A-Z]?)',                     # W- forms like W-2, W-8BEN
            r'(CT-\d+)',                         # CT forms
            r'(SS-\d+)',                         # SS forms
            r'(\d{3,}[A-Z]?-\w+)',               # e.g., 1099-MISC, 941-X, 706-GS(D)
            r'(\d{4,})\b(?![\d-])',             # 4+ digit forms like 1040, 8865 (but not years - check later)
            r'(Schedule\s+[A-Z])(?:\s|$|\()',   # Schedule A, Schedule B (followed by space, end, or parenthesis)
            # Add more specific patterns if needed
        ]

        # 3. Find all potential matches in the title
        potential_matches = []
        for pattern in patterns_to_find:
            potential_matches.extend(re.findall(pattern, original_title, re.IGNORECASE))

        # 4. Filter and clean the matches
        if potential_matches:
            cleaned_matches = set() # Use a set to handle duplicates automatically
            for match in potential_matches:
                # Handle tuples from patterns with multiple capture groups (like the first one)
                form_part = match[-1] if isinstance(match, tuple) else match
                form_part = form_part.strip()

                # Basic Validity Checks:
                # - Not purely numeric and looks like a year
                is_year = form_part.isdigit() and len(form_part) == 4 and form_part.startswith(('19', '20'))
                # - Not a common ignored word (adjust list as needed)
                ignore_words = {'and', 'for', 'the', 'instructions', 'general', 'return', 'tax', 'income', 'credit', 'federal'}
                # - Not empty
                if form_part and not is_year and form_part.lower() not in ignore_words:
                    cleaned_matches.add(form_part)

            form_numbers_found = sorted(list(cleaned_matches))
            logger.debug(f"Patterns found: {form_numbers_found} in '{original_title}'")

        # 5. Final check and join
        if form_numbers_found:
            form_number_final = ", ".join(form_numbers_found)
        else:
            form_number_final = None
            logger.warning(f"Could not reliably extract any form number from title: '{original_title}'. Skipping.")
            skipped_count += 1
            continue
        # --- End REVISED Parsing Logic V3 --- #

        # Database interaction uses 'form_number_final' and 'title'
        try:
            # Check if instruction with this *potentially comma-separated* form number already exists
            # NOTE: If multiple forms could map to different DB entries, this logic might need adjustment
            # For now, we assume the comma-separated string is the key
            existing_instruction = session.query(FormInstruction).filter_by(form_number=form_number_final).first()

            if existing_instruction:
                # Update if URL or title changed
                updated = False
                if existing_instruction.title != title:
                    existing_instruction.title = title
                    updated = True
                if existing_instruction.html_url != link_url: 
                    existing_instruction.html_url = link_url
                    updated = True
                if updated:
                   logger.debug(f"Updating existing record for form(s) {form_number_final}")
                   updated_count += 1
            else:
                # Create new instruction
                logger.debug(f"Creating new record for form(s) {form_number_final} with title '{title}'")
                new_instruction = FormInstruction(
                    title=title,         
                    form_number=form_number_final, # Use cleaned, potentially comma-separated number
                    html_url=link_url 
                )
                session.add(new_instruction)
                saved_count += 1

        except IntegrityError as e:
            session.rollback()
            logger.error(f"Integrity error processing form(s) '{form_number_final}' from title '{title}': {e}")
            skipped_count += 1
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error processing form(s) '{form_number_final}' from title '{title}': {e}")
            skipped_count += 1
            
    # Final commit
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit final changes: {e}")

    logger.info(f"Scraping finished. Saved: {saved_count}, Updated: {updated_count}, Skipped: {skipped_count}")

def main():
    """Main function to run the scraper"""
    print("Starting IRS instruction scraper...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Get database session
        session = get_session()
        
        # Scrape instructions
        scrape_irs_instructions(session, clear=args.clear, force=args.force)
        
        # Display first 10 items as a sample
        print("\nSample of extracted instructions:")
        sample_instructions = session.query(FormInstruction).limit(10).all()
        for i, instr in enumerate(sample_instructions):
            print(f"{i+1}. Form {instr.form_number}: {instr.html_url}")
        
        total_count = session.query(FormInstruction).count()
        if total_count > 10:
            print(f"... and {total_count - 10} more items in database")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the session
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    main() 