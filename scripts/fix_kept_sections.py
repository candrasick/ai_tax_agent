#!/usr/bin/env python
"""
Script to fix null core_text values in us_code_section_revised for sections marked as 'keep'.
This addresses a data issue where sections marked as 'keep' in section_history
did not properly copy their core_text from us_code_section.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_kept_sections():
    """
    Fixes null core_text values in us_code_section_revised for sections that were
    marked as 'keep' in section_history by copying text from us_code_section.
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
            # Find all sections that need fixing
            query = text("""
                WITH kept_sections AS (
                    SELECT DISTINCT orig_section_id, version_changed
                    FROM section_history
                    WHERE action = 'keep'
                )
                SELECT 
                    r.id as revised_id,
                    r.orig_section_id,
                    r.version,
                    s.core_text as original_text
                FROM us_code_section_revised r
                JOIN kept_sections k ON 
                    r.orig_section_id = k.orig_section_id 
                    AND r.version = k.version_changed
                JOIN us_code_section s ON r.orig_section_id = s.id
                WHERE r.core_text IS NULL
            """)
            
            results = session.execute(query)
            rows = results.fetchall()
            
            if not rows:
                logger.info("No sections found that need fixing")
                return
            
            logger.info(f"Found {len(rows)} sections to fix")
            
            # Update each section
            for row in rows:
                update_query = text("""
                    UPDATE us_code_section_revised
                    SET core_text = :core_text
                    WHERE id = :id
                """)
                
                try:
                    session.execute(update_query, {
                        'core_text': row.original_text,
                        'id': row.revised_id
                    })
                    logger.info(f"Fixed section {row.orig_section_id} version {row.version}")
                except Exception as e:
                    logger.error(f"Error updating section {row.orig_section_id}: {e}")
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
                WHERE h.action = 'keep' AND r.core_text IS NULL
            """)
            
            remaining_null = session.execute(verify_query).scalar()
            logger.info(f"Remaining sections with null core_text marked as 'keep': {remaining_null}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    fix_kept_sections() 