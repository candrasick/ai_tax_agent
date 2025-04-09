"""
Database session management for the AI Tax Agent.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Get the database URL from environment variable or use a default
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///tax_data.db')

# Create the engine
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

def get_session():
    """
    Get a new database session.
    
    Returns:
        Session: A new database session.
    """
    return Session() 