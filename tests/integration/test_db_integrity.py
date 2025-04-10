"""
Integration tests for database integrity requirements.
Tests that data in the database meets expected constraints and relationships.
"""

import os
import sys
import pytest
from sqlalchemy import func, select, text
from collections import Counter

# Ensure the main project directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import database models and session
from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session

def test_no_duplicate_section_numbers():
    """
    Test that there are no duplicate section_number values in the us_code_section table.
    
    This ensures integrity of the section numbers in the database while still allowing
    for valid section variants like 25, 25A, 25B, and 25C.
    """
    session = get_session()
    
    try:
        # Query all section numbers from the database
        query = select(UsCodeSection.section_number)
        result = session.execute(query).scalars().all()
        
        # Check if we have enough records to test
        total_sections = len(result)
        if total_sections < 10:
            pytest.skip(f"Not enough data to test (only {total_sections} sections found)")
        
        print(f"\nFound {total_sections} total section numbers in database")
        
        # Find duplicates using Counter
        section_counts = Counter(result)
        duplicates = {section: count for section, count in section_counts.items() if count > 1}
        
        # Output helpful debugging information if duplicates are found
        if duplicates:
            print(f"Found {len(duplicates)} duplicate section numbers:")
            for section, count in sorted(duplicates.items(), key=lambda x: (-x[1], x[0])):
                print(f"  Section {section}: {count} occurrences")
                
                # Get the actual records for this section to provide more detail
                dupe_records = session.query(UsCodeSection).filter(
                    UsCodeSection.section_number == section
                ).all()
                
                for idx, record in enumerate(dupe_records):
                    print(f"    [{idx+1}] ID: {record.id}, Title: {record.section_title}")
        
        # Assert that no duplicates were found
        assert not duplicates, f"Found {len(duplicates)} duplicate section numbers in the database"
        
        print("PASSED: No duplicate section numbers found in the database")
    
    finally:
        session.close()

def test_valid_section_numbers_format():
    """
    Test that section numbers in the database follow expected formatting patterns.
    
    Validates that section numbers follow one of these patterns:
    1. Standard format: digits followed by optional letters/special chars ("1", "25A", "163(j)")
    2. Range format: "1101 to 1103" or "54A to 54F"
    3. List format: "1246, 1247"
    """
    session = get_session()
    
    try:
        # Query all section numbers from the database
        query = select(UsCodeSection.section_number)
        result = session.execute(query).scalars().all()
        
        # Check if we have enough records to test
        total_sections = len(result)
        if total_sections < 10:
            pytest.skip(f"Not enough data to test (only {total_sections} sections found)")
        
        print(f"\nValidating format of {total_sections} section numbers")
        
        # Define a validation function that allows standard sections, ranges, and lists
        def is_valid_format(section_number):
            import re
            
            # Check for empty strings
            if not section_number:
                return False
                
            # Pattern 1: Standard section numbers (digits + optional letters/special chars)
            standard_pattern = r'^\d+[A-Za-z0-9().,-]*$'
            
            # Pattern 2: Range format ("X to Y")
            range_pattern = r'^\d+[A-Za-z0-9().-]* to \d+[A-Za-z0-9().-]*$'
            
            # Pattern 3: List format ("X, Y")
            list_pattern = r'^\d+[A-Za-z0-9().-]*(, \d+[A-Za-z0-9().-]*)+$'
            
            # Check if the section number matches any of the valid patterns
            if (re.match(standard_pattern, section_number) or 
                re.match(range_pattern, section_number) or 
                re.match(list_pattern, section_number)):
                return True
                
            # Special case for Unicode en-dash in section numbers like "1400L to 1400U–3"
            if "–" in section_number and re.sub(r'–', '-', section_number):
                # Try again with the en-dash replaced by hyphen
                normalized = section_number.replace("–", "-")
                return is_valid_format(normalized)
                
            return False
        
        # Filter out invalid formatted section numbers
        invalid_sections = [s for s in result if not is_valid_format(s)]
        
        # Output helpful debugging information for invalid sections
        if invalid_sections:
            print(f"Found {len(invalid_sections)} invalidly formatted section numbers:")
            for section in sorted(invalid_sections):
                print(f"  Invalid section: '{section}'")
        
        # Assert that no invalid section numbers were found
        assert not invalid_sections, f"Found {len(invalid_sections)} invalidly formatted section numbers"
        
        print("PASSED: All section numbers have valid format")
    
    finally:
        session.close()

def test_no_subsection_identifiers():
    """
    Test that section_number values don't contain subsection identifiers.
    
    This ensures that values like "1/a" or "143(a)" that represent subsections
    aren't stored as main section numbers.
    """
    session = get_session()
    
    try:
        # Query all section numbers from the database
        query = select(UsCodeSection.section_number)
        result = session.execute(query).scalars().all()
        
        # Check if we have enough records to test
        total_sections = len(result)
        if total_sections < 10:
            pytest.skip(f"Not enough data to test (only {total_sections} sections found)")
        
        print(f"\nChecking {total_sections} section numbers for subsection patterns")
        
        # Define patterns that would indicate a subsection rather than a main section
        def is_subsection_pattern(section_number):
            # Check for patterns like "1/a", "143/b/2", etc.
            if '/' in section_number:
                return True
                
            # Check for lowercase letter in parentheses like "143(a)"
            # This is a bit tricky since some legitimate section numbers might have parentheses
            # Example: 163(j) is a legitimate section number in the IRC
            # So we'll just check for lowercase letters in parentheses
            if '(' in section_number and ')' in section_number:
                inside_parens = section_number[section_number.find('(')+1:section_number.find(')')]
                if inside_parens.islower():
                    return True
            
            return False
        
        # Find section numbers with subsection patterns
        subsection_patterns = [s for s in result if is_subsection_pattern(s)]
        
        # Output helpful debugging information for subsection patterns
        if subsection_patterns:
            print(f"Found {len(subsection_patterns)} section numbers with subsection patterns:")
            for section in sorted(subsection_patterns):
                print(f"  Suspicious section: '{section}'")
        
        # Assert that no subsection patterns were found
        assert not subsection_patterns, f"Found {len(subsection_patterns)} section numbers with subsection patterns"
        
        print("PASSED: No section numbers contain subsection patterns")
    
    finally:
        session.close() 