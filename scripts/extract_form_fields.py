#!/usr/bin/env python
"""
Script to extract form field details from IRS form instructions and save them to the database.
Extracts content like "Line 1g. Other proceedings." and its explanatory text.

Options:
  --clear                 Clear existing form fields before extracting new ones
  --form FORM_NUMBER      Extract fields for a specific form number only
"""
import re
import requests
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# Import models and session from the project structure
from ai_tax_agent.database.models import FormInstruction, FormField
from ai_tax_agent.database.session import get_session

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('form_field_extractor')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Extract form fields from IRS instructions.")
    parser.add_argument('--clear', action='store_true', 
                        help='Clear existing form fields before extracting new ones')
    parser.add_argument('--form', type=str, help='Extract fields for a specific form number only')
    return parser.parse_args()

def extract_fields_from_html(html_content, form_number):
    """
    Extract field labels and their explanatory text from HTML content.
    Looks for patterns like "Line 1g. Other proceedings." followed by explanatory text.
    
    Args:
        html_content (str): HTML content of the form instructions
        form_number (str): Form number for logging purposes
        
    Returns:
        list: List of dictionaries with field_label and full_text keys
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all paragraphs in the document
    paragraphs = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # More flexible pattern to match various field labels
    # This handles cases like:
    # - "Line 1d." or "Line 1d"
    # - "Line 1g. Other proceedings."
    # - "Lines 1-4."
    # - "Line 1a, 1b, or 1c"
    field_pattern = re.compile(r'^(Line[s]?\s+(?:[\w\d]+[,.\s-]*)+(?:[^.]*\.)?)(.*)$', re.IGNORECASE)
    
    fields = []
    current_field = None
    current_text = []
    
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
        
        # Check if this paragraph starts with a field label
        match = field_pattern.match(text)
        
        if match:
            # If we were processing a previous field, save it
            if current_field and current_text:  # Only save if we have text content
                fields.append({
                    'field_label': current_field,
                    'full_text': ' '.join(current_text).strip()
                })
            
            # Start new field
            current_field = match.group(1).strip()
            if not current_field.endswith('.'):
                current_field += '.'
            
            # If there's text on the same line as the field label, include it
            if match.group(2).strip():
                current_text = [match.group(2).strip()]
            else:
                current_text = []
        elif current_field:
            # Add this paragraph to the current field's text if it's not a heading
            # (Headings often break the flow of field instructions)
            if not re.match(r'^[A-Z][a-z]*\s*[A-Z][a-z]*\.?$', text) and not p.name.startswith('h'):
                current_text.append(text)
            # If we hit what looks like a section heading, end the current field
            else:
                if current_text:  # Only save if we have text content
                    fields.append({
                        'field_label': current_field,
                        'full_text': ' '.join(current_text).strip()
                    })
                current_field = None
                current_text = []
    
    # Add the last field if we have one
    if current_field and current_text:  # Only save if we have text content
        fields.append({
            'field_label': current_field,
            'full_text': ' '.join(current_text).strip()
        })
    
    # Filter out fields with very short explanations (likely false positives)
    fields = [f for f in fields if len(f['full_text']) > 15]
    
    logger.info(f"Extracted {len(fields)} field labels from form {form_number}")
    return fields

def clear_form_fields(session, form_number=None):
    """
    Clear form fields from the database.
    
    Args:
        session: SQLAlchemy session
        form_number: If provided, only clear fields for this form number
    """
    if form_number:
        # Get the form instruction record
        instruction = session.query(FormInstruction).filter(
            FormInstruction.form_number.like(f"%{form_number}%")
        ).first()
        
        if not instruction:
            logger.error(f"Form {form_number} not found in database")
            return
        
        # Count fields for this form
        field_count = session.query(FormField).filter_by(
            instruction_id=instruction.id
        ).count()
        
        if field_count == 0:
            logger.info(f"No existing fields found for form {form_number}")
            return
        
        # Ask for confirmation
        should_continue = input(f"This will delete {field_count} fields for form {form_number}. Continue? (y/n): ").lower()
        if should_continue != 'y':
            logger.info("Operation cancelled.")
            return
        
        # Delete fields for this form
        session.query(FormField).filter_by(instruction_id=instruction.id).delete()
        session.commit()
        logger.info(f"Deleted {field_count} fields for form {form_number}")
    else:
        # Count all fields
        field_count = session.query(FormField).count()
        
        if field_count == 0:
            logger.info("No existing fields found in the database")
            return
        
        # Ask for confirmation
        should_continue = input(f"This will delete all {field_count} form fields from the database. Continue? (y/n): ").lower()
        if should_continue != 'y':
            logger.info("Operation cancelled.")
            return
        
        # Delete all fields
        session.query(FormField).delete()
        session.commit()
        logger.info(f"Deleted {field_count} form fields from the database")

def process_form_instruction(instruction, session, clear=False):
    """
    Process a single form instruction, extract fields, and save to database.
    
    Args:
        instruction (FormInstruction): The form instruction to process
        session: SQLAlchemy session
        clear: Whether to clear existing fields for this form
        
    Returns:
        int: Number of fields extracted and saved
    """
    logger.info(f"Processing form {instruction.form_number}")
    
    # Skip if no HTML URL
    if not instruction.html_url:
        logger.warning(f"No HTML URL for form {instruction.form_number}, skipping")
        return 0
    
    # Clear existing fields if requested
    if clear:
        clear_form_fields(session, instruction.form_number)
    
    # Fetch HTML content
    try:
        response = requests.get(instruction.html_url, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch HTML for form {instruction.form_number}: HTTP {response.status_code}")
            return 0
            
        # Extract fields
        fields = extract_fields_from_html(response.text, instruction.form_number)
        
        # Skip if no fields found
        if not fields:
            logger.warning(f"No fields found for form {instruction.form_number}")
            return 0
        
        # Save fields to database
        saved_count = 0
        updated_count = 0
        for field_data in fields:
            # Check if this field already exists
            existing = session.query(FormField).filter_by(
                instruction_id=instruction.id,
                field_label=field_data['field_label']
            ).first()
            
            if existing:
                # Update existing field
                existing.full_text = field_data['full_text']
                updated_count += 1
            else:
                # Create new field
                new_field = FormField(
                    instruction_id=instruction.id,
                    field_label=field_data['field_label'],
                    full_text=field_data['full_text']
                )
                session.add(new_field)
                saved_count += 1
        
        # Commit changes
        session.commit()
        logger.info(f"Saved {saved_count} new fields and updated {updated_count} existing fields for form {instruction.form_number}")
        
        return saved_count
        
    except Exception as e:
        logger.error(f"Error processing form {instruction.form_number}: {e}")
        return 0

def main():
    """Main function to run the field extractor"""
    logger.info("Starting IRS form field extractor...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Get database session
        session = get_session()
        
        # Clear all form fields if requested (and no specific form is provided)
        if args.clear and not args.form:
            clear_form_fields(session)
        
        # Get all form instructions or a specific form
        if args.form:
            instruction = session.query(FormInstruction).filter(
                FormInstruction.form_number.like(f"%{args.form}%")
            ).first()
            
            if not instruction:
                logger.error(f"Form {args.form} not found in database")
                return
                
            instructions = [instruction]
            logger.info(f"Processing specific form: {instruction.form_number}")
        else:
            instructions = session.query(FormInstruction).all()
            logger.info(f"Found {len(instructions)} form instructions in database")
            
            # If not processing a specific form, ask for confirmation before processing all forms
            if not args.form and not args.clear:
                should_continue = input(f"This will process {len(instructions)} forms. Continue? (y/n): ").lower()
                if should_continue != 'y':
                    logger.info("Operation cancelled.")
                    return
        
        # Process forms
        total_fields = 0
        
        for instruction in tqdm(instructions, desc="Processing forms"):
            fields_saved = process_form_instruction(instruction, session, clear=args.clear)
            total_fields += fields_saved
            
        logger.info(f"Processed {len(instructions)} forms, saved {total_fields} new fields")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Close the session
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    main() 