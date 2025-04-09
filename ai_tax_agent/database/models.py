from sqlalchemy.orm import declarative_base
from sqlalchemy import (Column, Integer, String, Text, Boolean, 
                        Date, TIMESTAMP, ForeignKey, func)

# Base class for all ORM models
Base = declarative_base()

# --- Define your database models below this line --- #

class UsCodeSection(Base):
    __tablename__ = 'us_code_section'

    id = Column(Integer, primary_key=True) # SERIAL/AUTOINCREMENT handled by default
    parent_id = Column(Integer, ForeignKey('us_code_section.id'), nullable=True)
    title_number = Column(Integer, nullable=True)
    subtitle = Column(Text, nullable=True)
    chapter = Column(Text, nullable=True)
    subchapter = Column(Text, nullable=True)
    part = Column(Text, nullable=True)
    subpart = Column(Text, nullable=True)
    section_number = Column(Text, nullable=False) # Using Text for flexibility (e.g., §280G, 1A)
    section_title = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    amendments_text = Column(Text, nullable=True) # Added for separated amendment notes
    core_text = Column(Text, nullable=True) # Added for core legal text without amendments
    amendment_count = Column(Integer, default=0, nullable=False)
    special_formatting = Column(Boolean, default=False, nullable=False)
    updated_at = Column(Date, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<UsCodeSection(id={self.id}, section_number='{self.section_number}', title='{self.section_title}')>"

# Example model (can be removed later):
# from sqlalchemy import Column, Integer, String
#
# class ExampleTable(Base):
#     __tablename__ = 'example_table'
#     id = Column(Integer, primary_key=True)
#     name = Column(String(50)) 