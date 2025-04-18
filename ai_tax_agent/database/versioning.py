#!/usr/bin/env python
"""Functions related to version tracking for tax code simplification."""

import logging
from typing import Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func

# Assuming project structure allows these imports relative to project root
# Adjust if your structure or execution context is different
try:
    from ai_tax_agent.database.session import get_session
    from ai_tax_agent.database.models import UsCodeSection, UsCodeSectionRevised
except ImportError:
    # Handle cases where script might be run directly and path isn't set up
    import sys
    import os
    # Add project root to sys.path (assuming this script is one level down)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai_tax_agent.database.session import get_session
    from ai_tax_agent.database.models import UsCodeSection, UsCodeSectionRevised

logger = logging.getLogger(__name__)
# Configure logging if run standalone, otherwise assume parent configures it
if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def determine_version_numbers() -> Tuple[int, int]:
    """
    Determines the prior and working version numbers for a simplification run.

    Looks at the `us_code_section_revised` table to find the highest version
    number and compares the count of rows for that version against the count
    in the `us_code_section` table.

    Returns:
        A tuple containing (prior_version, working_version).
    """
    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        # Consider raising an error or returning a specific failure indicator
        # Returning (0, 1) might be misleading if DB connection failed
        raise ConnectionError("Failed to establish database session for version check.")

    prior_version = 0
    working_version = 1

    try:
        logger.info("Determining latest existing revision version...")
        # Find the highest version number currently in the revised table
        max_version = db.query(
            sql_func.max(UsCodeSectionRevised.version)
        ).scalar()

        if max_version is None:
            # No revisions found, this is the first run.
            logger.info("No existing revisions found. Starting version 1.")
            prior_version = 0
            working_version = 1
        else:
            # Revisions exist, check if the latest version is complete
            logger.info(f"Highest existing revision version found: {max_version}")

            # Count total original sections
            orig_section_count = db.query(UsCodeSection.id).count()
            logger.debug(f"Total original sections count: {orig_section_count}")

            # Count sections in the highest existing revision
            revised_count_for_max = db.query(UsCodeSectionRevised.id)\
                .filter(UsCodeSectionRevised.version == max_version)\
                .count()
            logger.debug(f"Sections found in revision version {max_version}: {revised_count_for_max}")

            if revised_count_for_max >= orig_section_count:
                # The latest version is complete (or has at least as many rows).
                # Start a new version for the next run.
                logger.info(f"Version {max_version} appears complete ({revised_count_for_max} >= {orig_section_count}). Starting new version.")
                prior_version = max_version
                working_version = max_version + 1
            else:
                # The latest version is incomplete. Continue working on it.
                # The 'prior' version is the one before this incomplete one.
                logger.info(f"Version {max_version} is incomplete ({revised_count_for_max} < {orig_section_count}). Continuing work on version {max_version}.")
                prior_version = max(0, max_version - 1) # Ensure prior_version doesn't go below 0
                working_version = max_version

    except Exception as e:
        logger.error(f"Error during version determination: {e}", exc_info=True)
        # Re-raise or handle as appropriate for your application context
        raise
    finally:
        if db:
            db.close()
            logger.debug("Database session closed after version check.")

    logger.info(f"Determined versions: Prior={prior_version}, Working={working_version}")
    return prior_version, working_version

# Example usage block for testing
if __name__ == "__main__":
    print("Testing version determination...")
    try:
        p_version, w_version = determine_version_numbers()
        print(f"Result -> Prior Version: {p_version}, Working Version: {w_version}")
    except Exception as e:
        print(f"Test failed with error: {e}") 