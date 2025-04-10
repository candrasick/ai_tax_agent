import pytest
from sqlalchemy.orm import sessionmaker
from ai_tax_agent.database.models import FormInstruction, Base
from ai_tax_agent.database.session import get_session, engine
import logging

logger = logging.getLogger(__name__)

# Fixture to create a session for the test
@pytest.fixture(scope='module')
def db_session():
    """Provides a database session for the test module."""
    # Ensure the tables are created (optional, depending on setup)
    # Base.metadata.create_all(engine)
    session = get_session()
    yield session
    session.close()

def test_form_number_format(db_session):
    """Verify that form_number in FormInstruction does not contain parentheses."""
    logger.info("Testing form_number format in FormInstruction table...")
    
    invalid_forms = []
    all_instructions = db_session.query(FormInstruction).all()
    
    if not all_instructions:
        logger.warning("No FormInstruction records found in the database to test.")
        pytest.skip("No FormInstruction records found to test.")
        
    for instruction in all_instructions:
        if '(' in instruction.form_number or ')' in instruction.form_number:
            invalid_forms.append(instruction.form_number)
            
    if invalid_forms:
        logger.error(f"Found {len(invalid_forms)} form_numbers with invalid parentheses: {invalid_forms}")
    else:
        logger.info(f"Validated {len(all_instructions)} form numbers. No parentheses found.")
        
    assert not invalid_forms, f"Found form numbers containing parentheses: {invalid_forms}" 