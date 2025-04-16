# scripts/plot_complexity_distribution.py

import argparse
import logging
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session
from sqlalchemy import select

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings # Assuming settings might be needed indirectly by session
from ai_tax_agent.database.session import get_session
# Import relevant models
from ai_tax_agent.database.models import SectionComplexity

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = "complexity.png"

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate a histogram plot of section complexity scores.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # Reduce matplotlib verbosity
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    logger.info("--- Starting Complexity Plotting Script ---")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    scores: List[float] = []
    try:
        db: Session = get_session()
        logger.info("Database session acquired.")

        # Query SectionComplexity for scores
        logger.info("Querying database for complexity scores...")
        query = (
            select(SectionComplexity.complexity_score)
            .where(SectionComplexity.complexity_score.isnot(None)) # Only get non-null scores
        )

        results = db.execute(query).scalars().all() # Fetch all scores
        scores = [float(s) for s in results] # Ensure they are floats

        logger.info(f"Found {len(scores)} sections with complexity scores.")

    except Exception as e:
        logger.critical(f"An error occurred during database query: {e}", exc_info=True)
        if 'db' in locals() and db.is_active:
             db.rollback()
        return # Exit if DB query fails
    finally:
        if 'db' in locals() and db:
            db.close()
            logger.info("Database session closed.")

    if not scores:
        logger.warning("No complexity scores found in the database. Cannot generate plot.")
        return

    # --- Plotting ---
    try:
        logger.info("Generating complexity score distribution plot...")
        plt.style.use('seaborn-v0_8-deep') # Use a style similar to the example
        plt.figure(figsize=(14, 7)) # Set figure size similar to example

        # Create histogram with KDE
        sns.histplot(scores, kde=True, bins=30, edgecolor='black') # Adjust bins as needed

        # Set labels and title
        plt.xlabel("Complexity Score")
        plt.ylabel("Number of Sections")
        plt.title("Distribution of Complexity Scores Across All Sections")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Ensured output directory '{OUTPUT_DIR}' exists.")

        # Save the plot
        plt.savefig(output_path)
        logger.info(f"Plot saved successfully to {output_path}")
        plt.close() # Close the plot figure

    except Exception as e:
        logger.critical(f"An error occurred during plot generation or saving: {e}", exc_info=True)

    logger.info("--- Complexity Plotting Script Finished ---")


if __name__ == "__main__":
    main() 