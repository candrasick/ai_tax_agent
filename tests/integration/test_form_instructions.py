"""
Integration tests for the form instructions database storage.
Verifies that the form instruction titles are processed and stored correctly.
"""
import pytest
import sqlalchemy
from sqlalchemy.sql import text

from ai_tax_agent.database.models import FormInstruction
from ai_tax_agent.database.session import get_session


def test_form_instruction_titles():
    """
    Test that form instruction titles in the database:
    1. Don't contain the word "Instructions"
    2. Don't contain the word "Form"
    
    These words should be extracted and stored separately rather than
    being part of the title field.
    """
    # Get session
    session = get_session()
    
    try:
        # Get all form instructions
        instructions = session.query(FormInstruction).all()
        
        # Ensure we have at least some data to test
        assert len(instructions) > 0, "No form instructions found in database. Run 'make scrape-instructions' first."
        
        # Check each instruction title for the forbidden words
        invalid_titles = []
        
        for instruction in instructions:
            title = instruction.title
            
            if "Instructions" in title:
                invalid_titles.append({
                    "id": instruction.id,
                    "form_number": instruction.form_number,
                    "title": title,
                    "issue": "Contains 'Instructions'"
                })
                
            if "Form" in title:
                invalid_titles.append({
                    "id": instruction.id,
                    "form_number": instruction.form_number,
                    "title": title,
                    "issue": "Contains 'Form'"
                })
        
        # Display detailed information about invalid titles
        if invalid_titles:
            error_message = "\nInvalid form instruction titles found:\n"
            for item in invalid_titles[:10]:  # Show at most 10 examples
                error_message += f"  - ID {item['id']}, Form {item['form_number']}: '{item['title']}' ({item['issue']})\n"
            
            if len(invalid_titles) > 10:
                error_message += f"  - ... and {len(invalid_titles) - 10} more invalid titles\n"
                
            error_message += "\nForm instruction titles should not contain 'Instructions' or 'Form'. "
            error_message += "These words should be extracted and stored in the form_number field."
            
            pytest.fail(error_message)
    
    finally:
        # Clean up
        session.close()


def test_form_number_extraction():
    """
    Test that form numbers are correctly extracted from original titles
    and don't include any unnecessary words like 'Form' or 'Instructions'.
    """
    session = get_session()
    
    try:
        # Get all form instructions
        instructions = session.query(FormInstruction).all()
        
        # Ensure we have at least some data to test
        assert len(instructions) > 0, "No form instructions found in database. Run 'make scrape-instructions' first."
        
        # Check form numbers
        invalid_form_numbers = []
        
        for instruction in instructions:
            form_num = instruction.form_number
            
            # Form numbers should not include "Instructions" or "Form"
            if "Instructions" in form_num:
                invalid_form_numbers.append({
                    "id": instruction.id,
                    "form_number": form_num,
                    "issue": "Contains 'Instructions'"
                })
                
            if "Form" in form_num:
                invalid_form_numbers.append({
                    "id": instruction.id,
                    "form_number": form_num,
                    "issue": "Contains 'Form'"
                })
        
        # Display detailed information about invalid form numbers
        if invalid_form_numbers:
            error_message = "\nInvalid form numbers found:\n"
            for item in invalid_form_numbers[:10]:  # Show at most 10 examples
                error_message += f"  - ID {item['id']}: '{item['form_number']}' ({item['issue']})\n"
            
            if len(invalid_form_numbers) > 10:
                error_message += f"  - ... and {len(invalid_form_numbers) - 10} more invalid form numbers\n"
                
            error_message += "\nForm numbers should not contain 'Instructions' or 'Form'."
            
            pytest.fail(error_message)
    
    finally:
        # Clean up
        session.close()


if __name__ == "__main__":
    test_form_instruction_titles()
    test_form_number_extraction() 