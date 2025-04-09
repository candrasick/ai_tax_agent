"""
Database session management for the AI Tax Agent.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the settings instance
from ai_tax_agent.settings import settings

# Get the database URL from settings
DATABASE_URL = settings.database_url

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