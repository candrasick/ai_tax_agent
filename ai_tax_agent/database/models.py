from sqlalchemy.orm import declarative_base
from sqlalchemy import (Column, Integer, String, Text, Boolean, 
                        Date, TIMESTAMP, ForeignKey, func, UniqueConstraint)
from sqlalchemy.orm import relationship

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
    core_text = Column(Text, nullable=True) # Added for main section text without amendments
    amendment_count = Column(Integer, default=0, nullable=True)
    special_formatting = Column(Boolean, default=False)
    updated_at = Column(Date, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    # Add unique constraint for section_number
    __table_args__ = (UniqueConstraint('section_number', name='uq_section_number'),)

    # Relationship to Form Fields (many-to-many through FormFieldUsCodeSectionLink)
    field_links = relationship("FormFieldUsCodeSectionLink", back_populates="us_code_section", cascade="all, delete-orphan")
    # Relationship to bulletin items (existing)
    bulletin_item_associations = relationship("IrsBulletinItemToCodeSection", back_populates="us_code_section")
    
    def __repr__(self):
        return f"<UsCodeSection(id={self.id}, section_number='{self.section_number}', title='{self.section_title}')>"

# Example model (can be removed later):
# from sqlalchemy import Column, Integer, String
#
# class ExampleTable(Base):
#     __tablename__ = 'example_table'
#     id = Column(Integer, primary_key=True)
#     name = Column(String(50)) 

class IrsBulletin(Base):
    __tablename__ = 'irs_bulletin'

    id = Column(Integer, primary_key=True)
    bulletin_number = Column(Text, nullable=False) # e.g., '2024-08'
    bulletin_date = Column(Date, nullable=True)
    title = Column(Text, nullable=True)
    source_url = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    # Relationship to items within this bulletin
    items = relationship("IrsBulletinItem", back_populates="bulletin", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<IrsBulletin(id={self.id}, number='{self.bulletin_number}')>"


class IrsBulletinItem(Base):
    __tablename__ = 'irs_bulletin_item'

    id = Column(Integer, primary_key=True)
    bulletin_id = Column(Integer, ForeignKey('irs_bulletin.id'), nullable=False)
    # Documenting potential values, CHECK constraint might need DB-specific implementation or app logic
    item_type = Column(Text, nullable=True) # 'Rev. Rul.', 'Rev. Proc.', 'Announcement', 'Notice', 'Disbarment', 'Other'
    item_number = Column(Text, nullable=False) # e.g., '2024-12'
    title = Column(Text, nullable=True)
    action = Column(Text, nullable=True) # Comma-separated actions like 'modified', 'superseded'
    full_text = Column(Text, nullable=True) # Full text snippet for the item
    referenced_sections = Column(Text, nullable=True) # Comma-separated list of section numbers found
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    # Relationship back to the parent bulletin
    bulletin = relationship("IrsBulletin", back_populates="items")
    # Relationship to linked code sections
    code_section_associations = relationship("IrsBulletinItemToCodeSection", back_populates="bulletin_item", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<IrsBulletinItem(id={self.id}, type='{self.item_type}', number='{self.item_number}')>"


class IrsBulletinItemToCodeSection(Base):
    __tablename__ = 'irs_bulletin_item_to_code_section'

    id = Column(Integer, primary_key=True)
    bulletin_item_id = Column(Integer, ForeignKey('irs_bulletin_item.id'), nullable=False)
    section_id = Column(Integer, ForeignKey('us_code_section.id'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    # Define composite unique constraint
    __table_args__ = (UniqueConstraint('bulletin_item_id', 'section_id', name='uq_bulletin_item_section'),)

    # Relationships
    bulletin_item = relationship("IrsBulletinItem", back_populates="code_section_associations")
    us_code_section = relationship("UsCodeSection", back_populates="bulletin_item_associations")

    def __repr__(self):
        return f"<IrsBulletinItemToCodeSection(item_id={self.bulletin_item_id}, section_id={self.section_id})>"

class FormInstruction(Base):
    """Model for IRS form instructions"""
    __tablename__ = 'form_instructions'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    form_number = Column(String(50), nullable=False, index=True)
    html_url = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    
    # Relationship to form fields
    fields = relationship("FormField", back_populates="instruction")
    
    def __repr__(self):
        return f"<FormInstruction(form_number='{self.form_number}')>"


class FormField(Base):
    """Model for form instruction fields"""
    __tablename__ = 'form_fields'
    
    id = Column(Integer, primary_key=True)
    instruction_id = Column(Integer, ForeignKey('form_instructions.id', ondelete='CASCADE'), nullable=False, index=True)
    field_label = Column(String(100), nullable=False)  # e.g., "Line 1g. Other proceedings."
    full_text = Column(Text, nullable=False)  # The complete instruction text for this field
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    
    # Relationship to parent instruction
    instruction = relationship("FormInstruction", back_populates="fields")
    
    def __repr__(self):
        return f"<FormField(field_label='{self.field_label}')>"

# --- Association Object Model: FormField <-> UsCodeSection --- #
class FormFieldUsCodeSectionLink(Base):
    __tablename__ = 'form_field_us_code_section_link'
    id = Column(Integer, primary_key=True) # Surrogate primary key
    form_field_id = Column(Integer, ForeignKey('form_fields.id', ondelete='CASCADE'), nullable=False, index=True)
    us_code_section_id = Column(Integer, ForeignKey('us_code_section.id', ondelete='CASCADE'), nullable=False, index=True)
    rationale = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    
    # Add unique constraint to prevent duplicate links for the same pair
    __table_args__ = (UniqueConstraint('form_field_id', 'us_code_section_id', name='uq_form_field_section_link'),)

    # Relationships back to the parent tables
    form_field = relationship("FormField", back_populates="section_links")
    us_code_section = relationship("UsCodeSection", back_populates="field_links")

    def __repr__(self):
        return f"<FormFieldUsCodeSectionLink(form_field_id={self.form_field_id}, us_code_section_id={self.us_code_section_id})>" 