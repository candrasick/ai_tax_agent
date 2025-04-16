# scripts/export_complexity_data.py

import argparse
import logging
import os
import sys
import json
from typing import List, Dict, Any, Sequence

from sqlalchemy.orm import Session
from sqlalchemy import select

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings # Assuming settings might be needed indirectly by session
from ai_tax_agent.database.session import get_session
# Import relevant models
from ai_tax_agent.database.models import UsCodeSection, SectionComplexity

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "complexity.json"

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Export US Code section complexity data to JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)
    logger.info("--- Starting Complexity Data Export Script ---")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    try:
        db: Session = get_session()
        logger.info("Database session acquired.")

        # Query UsCodeSection joined with SectionComplexity
        # This implicitly performs an INNER JOIN, only including sections with complexity scores
        logger.info("Querying database for sections with complexity scores...")
        query = (
            select(
                UsCodeSection.subtitle,
                UsCodeSection.chapter,
                UsCodeSection.subchapter,
                UsCodeSection.part,
                UsCodeSection.section_number,
                UsCodeSection.section_title,
                SectionComplexity.complexity_score,
                SectionComplexity.rationale
            )
            .join(SectionComplexity, UsCodeSection.id == SectionComplexity.section_id)
            .order_by(
                # Order predictably, e.g., by title hierarchy then section number
                # Note: Need robust sorting if section_number isn't purely numeric
                UsCodeSection.title_number.nullslast(),
                UsCodeSection.section_number # Assuming section_number allows reasonable sorting
            )
        )

        results = db.execute(query).all() # Fetch all results

        logger.info(f"Found {len(results)} sections with complexity data.")

        if not results:
            logger.warning("No complexity data found in the database. Output file will be empty.")
            output_data = []
        else:
            # Format data for JSON output
            output_data: List[Dict[str, Any]] = []
            for row in results:
                output_data.append({
                    "subtitle": row.subtitle,
                    "chapter": row.chapter,
                    "subchapter": row.subchapter,
                    "part": row.part,
                    "section_number": row.section_number,
                    "section_title": row.section_title,
                    "complexity_score": row.complexity_score,
                    "rationale": row.rationale
                })

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Ensured output directory '{OUTPUT_DIR}' exists.")

        # Write data to JSON file
        logger.info(f"Writing {len(output_data)} records to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully wrote data to {output_path}")

    except Exception as e:
        logger.critical(f"An error occurred during the export process: {e}", exc_info=True)
        if 'db' in locals() and db.is_active:
            db.rollback() # Just in case, though selects shouldn't need it
    finally:
        if 'db' in locals() and db:
            db.close()
            logger.info("Database session closed.")

    logger.info("--- Complexity Data Export Script Finished ---")


if __name__ == "__main__":
    main() 