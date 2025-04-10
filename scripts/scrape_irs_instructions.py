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

# Import models and session from the project structure
from ai_tax_agent.database.models import FormInstruction, FormField
from ai_tax_agent.database.session import get_session

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Scrape IRS form instructions.")
    parser.add_argument('--clear', action='store_true', 
                        help='Clear existing form instructions before scraping new ones')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompts')
    return parser.parse_args()

def clean_text(text):
    """
    Remove 'Instructions', 'Form', and clean up extra spaces.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Handle special case for "Schedule LEP Instructions"
    if "Schedule LEP Instructions" in text:
        return "Schedule LEP"
        
    # Remove date/version in parentheses to handle them separately
    version_info = ""
    date_match = re.search(r'\((\d{2}/\d{4}|\d{2}/\d{2}/\d{4}|\d{4})\)$', text)
    if date_match:
        version_info = date_match.group(0)
        text = text.replace(version_info, "").strip()
    
    # Remove "Instructions for"
    cleaned = text.replace("Instructions for", "").strip()
    
    # Handle "Forms X and Y" pattern
    forms_match = re.search(r'Forms\s+(\d+[A-Z0-9-]+)\s+and\s+(\d+[A-Z0-9-]+)', cleaned)
    if forms_match:
        form1 = forms_match.group(1).strip()
        form2 = forms_match.group(2).strip()
        cleaned = f"{form1} and {form2}"
    
    # Handle schedules with forms in a special way
    elif re.search(r'Schedule\s+[A-Z0-9-]+\s+\(Form\s+\d+.*?\)', cleaned):
        # Convert "Schedule X (Form Y)" to "Schedule X-Y"
        match = re.search(r'(Schedule\s+[A-Z0-9-]+)\s+\(Form\s+(\d+[^)]*)\)', cleaned)
        if match:
            schedule = match.group(1).strip()
            form_num = match.group(2).strip()
            cleaned = f"{schedule}-{form_num}"
            
    # Handle "Partner's Schedule K-1 (Form 1065)" type cases
    elif "Schedule" in cleaned and "(Form" in cleaned:
        match = re.search(r'(.*Schedule\s+[A-Z0-9-]+)\s+\(Form\s+(\d+[^)]*)\)', cleaned)
        if match:
            prefix = match.group(1).strip()
            form_num = match.group(2).strip()
            cleaned = f"{prefix}-{form_num}"
            
    # Handle "Partnership Schedules K-2 and K-3 (Form 1065)" type cases
    elif "Schedules" in cleaned and "(Form" in cleaned:
        match = re.search(r'(.*Schedules\s+[A-Z0-9-]+\s+and\s+[A-Z0-9-]+)\s+\(Form\s+(\d+[^)]*)\)', cleaned)
        if match:
            prefix = match.group(1).strip()
            form_num = match.group(2).strip()
            cleaned = f"{prefix}-{form_num}"
    
    # Now remove all remaining instances of "Form" that aren't part of a more complex pattern
    cleaned = re.sub(r'\bForm\b', '', cleaned)
    
    # Also remove plural "Forms"
    cleaned = re.sub(r'\bForms\b', '', cleaned)
    
    # Remove "Instructions" anywhere in the text
    cleaned = cleaned.replace("Instructions", "").strip()
    
    # Fix multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove "for" at the beginning that might be left over
    if cleaned.startswith("for "):
        cleaned = cleaned[4:].strip()
    
    # Remove any parentheses with nothing inside them
    cleaned = re.sub(r'\(\s*\)', '', cleaned).strip()
    
    # Reattach the version info if it was present
    if version_info and version_info not in cleaned:
        cleaned = cleaned + " " + version_info
        
    # Final cleanup of multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
    return cleaned

def scrape_irs_instructions():
    """
    Scrape the IRS instructions page and extract form numbers and links.
    Returns a list of dictionaries with form information.
    """
    print("Fetching IRS instructions page...")
    url = "https://www.irs.gov/instructions"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch IRS instructions page: {response.status_code}")
    
    print("Parsing HTML content...")
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main table containing instruction listings
    tables = soup.find_all('table')
    
    if not tables:
        raise Exception("Could not find instruction table on the page")
    
    main_table = tables[0]  # Assuming the first table contains the instructions
    
    # Extract rows from the table
    rows = main_table.find_all('tr')
    
    # Skip header row
    instructions = []
    
    print("Extracting instruction information...")
    for row in tqdm(rows[1:], desc="Processing instructions"):
        cells = row.find_all('td')
        if len(cells) >= 3:
            title_cell = cells[0]
            html_link_cell = cells[1]
            # We'll ignore the PDF cell
            
            original_title = title_cell.text.strip()
            
            # Skip Spanish instructions (those with "Instrucciones" in the title)
            if original_title.startswith("Instrucciones"):
                continue
            
            # Clean the text to get both form number and title
            cleaned_text = clean_text(original_title)
            
            # Extract HTML link
            html_link = html_link_cell.find('a')
            if html_link:
                html_url = "https://www.irs.gov" + html_link['href'] if html_link['href'].startswith('/') else html_link['href']
            else:
                html_url = ""
            
            # Skip entries without HTML links
            if not html_url:
                continue
            
            instructions.append({
                'title': cleaned_text,
                'form_number': cleaned_text,
                'html_url': html_url,
            })
    
    print(f"Found {len(instructions)} English form instructions")
    return instructions


def clear_existing_data(session, force=False):
    """
    Clear all existing form instructions and related fields from the database.
    
    Args:
        session: SQLAlchemy session
        force: Whether to skip confirmation prompts
    """
    print("Clearing existing form instructions and fields...")
    
    # Get count of existing records
    instruction_count = session.query(FormInstruction).count()
    field_count = session.query(FormField).count()
    
    if instruction_count == 0 and field_count == 0:
        print("No existing data to clear.")
        return
    
    # Ask for confirmation unless force is True
    if not force:
        should_continue = input(f"This will delete {instruction_count} form instructions and {field_count} form fields. Continue? (y/n): ").lower()
        if should_continue != 'y':
            print("Operation cancelled.")
            return
    else:
        print(f"Force mode: Deleting {instruction_count} form instructions and {field_count} form fields without confirmation.")
    
    # Delete existing records
    # FormField records will be deleted automatically due to CASCADE
    session.query(FormInstruction).delete()
    session.commit()
    
    print(f"Deleted {instruction_count} form instructions and {field_count} form fields.")


def save_to_database(instructions, session, clear=False, force=False):
    """
    Save the extracted instructions to the database
    
    Args:
        instructions: List of instruction dictionaries
        session: SQLAlchemy session
        clear: Whether to clear existing data before saving
        force: Whether to skip confirmation prompts
    """
    print("Saving instructions to database...")
    
    # Clear existing data if requested
    if clear:
        clear_existing_data(session, force=force)
    
    # Count existing records
    existing_count = session.query(FormInstruction).count()
    if existing_count > 0 and not clear:
        print(f"Database already contains {existing_count} form instructions.")
        
        if not force:
            print("Performing incremental update (will add new forms and update existing ones).")
        else:
            print("Force mode: Performing incremental update without confirmation.")
    
    # Add new records
    new_records = 0
    updated_records = 0
    
    for instr in tqdm(instructions, desc="Saving to database"):
        # Check if this form already exists
        existing = session.query(FormInstruction).filter_by(form_number=instr['form_number']).first()
        if existing:
            # Update existing record
            existing.title = instr['title']
            existing.html_url = instr['html_url']
            updated_records += 1
        else:
            # Create new record
            new_form = FormInstruction(
                title=instr['title'],
                form_number=instr['form_number'],
                html_url=instr['html_url'],
            )
            session.add(new_form)
            new_records += 1
    
    # Commit changes
    session.commit()
    print(f"Saved {new_records} new instructions and updated {updated_records} existing instructions")


def main():
    """Main function to run the scraper"""
    print("Starting IRS instruction scraper...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Get database session
        session = get_session()
        
        # Scrape instructions
        instructions = scrape_irs_instructions()
        
        # Save to database
        save_to_database(instructions, session, clear=args.clear, force=args.force)
        
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