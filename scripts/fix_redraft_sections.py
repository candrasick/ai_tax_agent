#!/usr/bin/env python
"""
Script to fix sections marked for redraft that have null core_text.
This script directly uses the redraft_tool to process the original text,
bypassing the agent decision-making process.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from ai_tax_agent.tools.generation_tools import redraft_section_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_redraft_sections():
    """
    Fixes sections that were marked for redraft but have null core_text
    by applying the redraft_tool directly to their original text.
    """
    # Load environment variables
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not found in environment variables")
        return

    # Create database engine
    engine = create_engine(database_url)
    
    try:
        with Session(engine) as session:
            # Find all redraft sections that need fixing
            query = text("""
                WITH redraft_sections AS (
                    SELECT DISTINCT orig_section_id, version_changed
                    FROM section_history
                    WHERE action = 'redraft'
                    AND version_changed = 1
                )
                SELECT 
                    r.id as revised_id,
                    r.orig_section_id,
                    r.version,
                    s.core_text as original_text
                FROM us_code_section_revised r
                JOIN redraft_sections k ON 
                    r.orig_section_id = k.orig_section_id 
                    AND r.version = k.version_changed
                JOIN us_code_section s ON r.orig_section_id = s.id
                WHERE r.core_text IS NULL
                ORDER BY r.orig_section_id
            """)
            
            results = session.execute(query)
            rows = results.fetchall()
            
            if not rows:
                logger.info("No redraft sections found that need fixing")
                return
            
            logger.info(f"Found {len(rows)} redraft sections to fix")
            
            # Process each section
            for row in rows:
                if not row.original_text:
                    logger.warning(f"Section {row.orig_section_id} has no original text to redraft")
                    continue

                logger.info(f"Processing section {row.orig_section_id}")
                
                try:
                    # Apply redraft tool directly
                    redrafted_text = redraft_section_text(
                        section_text=row.original_text,
                        model_name="gemini-1.5-flash-latest",
                        temperature=0.2
                    )
                    
                    if redrafted_text.startswith("[Error:"):
                        logger.error(f"Redraft tool error for section {row.orig_section_id}: {redrafted_text}")
                        continue

                    # Update the section with the redrafted text
                    update_query = text("""
                        UPDATE us_code_section_revised
                        SET core_text = :core_text
                        WHERE id = :id
                    """)
                    
                    session.execute(update_query, {
                        'core_text': redrafted_text,
                        'id': row.revised_id
                    })
                    logger.info(f"Updated section {row.orig_section_id} with redrafted text")
                    
                except Exception as e:
                    logger.error(f"Error processing section {row.orig_section_id}: {e}")
                    session.rollback()
                    continue
            
            # Commit all changes
            try:
                session.commit()
                logger.info("Successfully committed all changes")
            except Exception as e:
                logger.error(f"Error committing changes: {e}")
                session.rollback()
                return
            
            # Verify the fix
            verify_query = text("""
                SELECT COUNT(*)
                FROM us_code_section_revised r
                JOIN section_history h ON 
                    r.orig_section_id = h.orig_section_id 
                    AND r.version = h.version_changed
                WHERE h.action = 'redraft' 
                AND h.version_changed = 1
                AND r.core_text IS NULL
            """)
            
            remaining_null = session.execute(verify_query).scalar()
            logger.info(f"Remaining sections with null core_text marked as 'redraft': {remaining_null}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    fix_redraft_sections() 