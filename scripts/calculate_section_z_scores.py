# scripts/calculate_section_z_scores.py

import argparse
import logging
import math
import os
import sys
from typing import List, Sequence, Tuple
import statistics # Using standard library for mean/stdev

from sqlalchemy.orm import Session, selectinload
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings # Assuming settings might be needed indirectly by session
from ai_tax_agent.database.session import get_session
# Import relevant models
from ai_tax_agent.database.models import UsCodeSection

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def calculate_stats(data: List[float]) -> Tuple[float, float]:
    """Calculates mean and standard deviation for a list of numbers."""
    if not data:
        return 0.0, 0.0 # Or handle as error?
    
    mean_val = statistics.mean(data)
    
    # Standard deviation requires at least 2 data points
    if len(data) < 2:
        std_dev_val = 0.0
    else:
        try:
            std_dev_val = statistics.stdev(data)
        except statistics.StatisticsError:
             # This can happen if all values are the same, stdev handles this mostly
             std_dev_val = 0.0

    # Handle case where stdev is exactly 0 (all values are the same)
    if std_dev_val == 0.0:
        logger.debug("Standard deviation is 0. All values are likely identical.")
        # Keep std_dev_val as 0.0, Z-scores will be 0.

    return mean_val, std_dev_val

def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """Calculates the Z-score, handling division by zero."""
    if std_dev == 0:
        # If std_dev is 0, all values are the mean, so z-score is 0
        return 0.0
    try:
        return (value - mean) / std_dev
    except ZeroDivisionError:
         # Should be caught by the std_dev == 0 check, but as safety
        return 0.0

# --- Main Calculation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Calculate and populate Z-score metrics for UsCodeSection table.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)
    logger.info("--- Starting Z-Score Calculation Script ---")

    try:
        db: Session = get_session()
        logger.info("Database session acquired.")

        # 1. Fetch all sections and related data efficiently
        logger.info("Fetching all UsCodeSection records and bulletin associations...")
        sections: Sequence[UsCodeSection] = db.query(UsCodeSection).options(
            selectinload(UsCodeSection.bulletin_item_associations) # Efficiently load related counts
        ).order_by(UsCodeSection.id).all()

        if not sections:
            logger.warning("No UsCodeSection records found in the database. Exiting.")
            db.close()
            return

        logger.info(f"Fetched {len(sections)} sections.")

        # 2. Extract metric values for all sections
        logger.info("Extracting metric values (amendment count, bulletin count, section length)...")
        amendment_counts: List[float] = []
        bulletin_counts: List[float] = []
        section_lengths: List[float] = []

        for section in tqdm(sections, desc="Extracting Metrics"):
            # Amendment Count
            amendment_counts.append(float(section.amendment_count or 0))

            # Bulletin Count (using loaded relationship)
            bulletin_counts.append(float(len(section.bulletin_item_associations)))

            # Section Length (core_text)
            section_lengths.append(float(len(section.core_text or "")))

        # 3. Calculate overall statistics (Mean, Standard Deviation)
        logger.info("Calculating overall mean and standard deviation for each metric...")
        
        mean_amendments, std_dev_amendments = calculate_stats(amendment_counts)
        logger.info(f"Amendment Counts: Mean={mean_amendments:.2f}, StdDev={std_dev_amendments:.2f}")

        mean_bulletins, std_dev_bulletins = calculate_stats(bulletin_counts)
        logger.info(f"Bulletin Counts: Mean={mean_bulletins:.2f}, StdDev={std_dev_bulletins:.2f}")
        
        mean_length, std_dev_length = calculate_stats(section_lengths)
        logger.info(f"Section Lengths: Mean={mean_length:.2f}, StdDev={std_dev_length:.2f}")

        # 4. Calculate and update Z-scores for each section
        logger.info("Calculating and updating Z-scores for each section...")
        update_count = 0
        for i, section in enumerate(tqdm(sections, desc="Updating Z-Scores")):
            try:
                # Get the individual values (already extracted)
                amendment_count = amendment_counts[i]
                bulletin_count = bulletin_counts[i]
                section_length = section_lengths[i]

                # Calculate Z-scores
                section.amendment_count_z = calculate_z_score(amendment_count, mean_amendments, std_dev_amendments)
                section.bulletins_count_z = calculate_z_score(bulletin_count, mean_bulletins, std_dev_bulletins)
                section.section_count_z = calculate_z_score(section_length, mean_length, std_dev_length)

                # Add to session (SQLAlchemy tracks changes)
                db.add(section)
                update_count += 1
                
                # Optional: Commit in batches for very large tables
                # if (i + 1) % 1000 == 0:
                #     logger.debug(f"Committing batch at index {i}...")
                #     db.commit()

            except Exception as e_inner:
                 logger.error(f"Error calculating/updating Z-score for section ID {section.id}: {e_inner}", exc_info=True)
                 db.rollback() # Rollback changes for this specific section if error occurs
                 # Decide if you want to continue with others or stop

        # 5. Commit final changes
        logger.info(f"Committing updates for {update_count} sections...")
        db.commit()
        logger.info("Updates committed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Z-score calculation process: {e}", exc_info=True)
        if 'db' in locals() and db.is_active:
            db.rollback()
    finally:
        if 'db' in locals() and db:
            db.close()
            logger.info("Database session closed.")

    logger.info("--- Z-Score Calculation Script Finished ---")


if __name__ == "__main__":
    main() 