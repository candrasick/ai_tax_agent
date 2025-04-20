#!/usr/bin/env python
"""Functions related to version tracking for tax code simplification."""

import logging
from typing import Tuple, Optional
from decimal import Decimal # Import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func, select, text

# Assuming project structure allows these imports relative to project root
# Adjust if your structure or execution context is different
try:
    from ai_tax_agent.database.session import get_session
    from ai_tax_agent.database.models import UsCodeSection, UsCodeSectionRevised, SectionImpact
except ImportError:
    # Handle cases where script might be run directly and path isn't set up
    import sys
    import os
    # Add project root to sys.path (assuming this script is one level down)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai_tax_agent.database.session import get_session
    from ai_tax_agent.database.models import UsCodeSection, UsCodeSectionRevised, SectionImpact

logger = logging.getLogger(__name__)
# Configure logging if run standalone, otherwise assume parent configures it
if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')

def determine_version_numbers() -> Tuple[int, int]:
    """
    Determines the prior and working version numbers for a simplification run.

    Looks at the `us_code_section_revised` table to find the highest version
    number and compares the count of rows for that version against the count
    in the `us_code_section` table, accounting for sections deleted in prior versions.

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

            # Get count of sections deleted in prior versions
            deleted_sections_count = db.query(UsCodeSectionRevised.orig_section_id)\
                .filter(UsCodeSectionRevised.deleted == True)\
                .filter(UsCodeSectionRevised.version < max_version)\
                .distinct()\
                .count()
            logger.debug(f"Sections deleted in prior versions: {deleted_sections_count}")

            # Count sections in the highest existing revision
            revised_count_for_max = db.query(UsCodeSectionRevised.id)\
                .filter(UsCodeSectionRevised.version == max_version)\
                .count()
            logger.debug(f"Sections found in revision version {max_version}: {revised_count_for_max}")

            # Total expected sections is original count minus previously deleted sections
            expected_section_count = orig_section_count - deleted_sections_count
            logger.debug(f"Expected sections to process: {expected_section_count}")

            if revised_count_for_max >= expected_section_count:
                # The latest version is complete (accounting for previously deleted sections)
                logger.info(f"Version {max_version} appears complete ({revised_count_for_max} >= {expected_section_count}). Starting new version.")
                prior_version = max_version
                working_version = max_version + 1
            else:
                # The latest version is incomplete. Continue working on it.
                # The 'prior' version is the one before this incomplete one.
                logger.info(f"Version {max_version} is incomplete ({revised_count_for_max} < {expected_section_count}). Continuing work on version {max_version}.")
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

def get_total_text_length_for_version(version_number: int) -> Optional[int]:
    """
    Calculates the total character length of the core_text for a given version.

    Args:
        version_number: The version number to calculate length for.
                        Version 0 refers to the original `us_code_section` table.
                        Versions 1+ refer to the `us_code_section_revised` table.

    Returns:
        The total character length as an integer, or None if an error occurs.
    """
    if version_number < 0:
        logger.error("Version number cannot be negative.")
        return None

    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        return None

    total_length: Optional[int] = None
    try:
        if version_number == 0:
            logger.info("Calculating total core_text length for original sections (version 0)...")
            # Use LENGTH function (SQLite/PostgreSQL standard). Adapt if needed for other DBs.
            # COALESCE handles NULL core_text values, treating them as length 0.
            length_query = db.query(
                sql_func.sum(sql_func.length(sql_func.coalesce(UsCodeSection.core_text, '')))
            )
            total_length = length_query.scalar()
        else:
            logger.info(f"Calculating total core_text length for revised version {version_number}...")
            length_query = db.query(
                sql_func.sum(sql_func.length(sql_func.coalesce(UsCodeSectionRevised.core_text, '')))
            ).filter(UsCodeSectionRevised.version == version_number)
            total_length = length_query.scalar()

        # Ensure result is an integer, default to 0 if None (e.g., empty table/version)
        total_length = int(total_length) if total_length is not None else 0
        logger.info(f"Calculated total length for version {version_number}: {total_length:,}")

    except Exception as e:
        logger.error(f"Error calculating text length for version {version_number}: {e}", exc_info=True)
        total_length = None # Ensure None is returned on error
    finally:
        if db:
            db.close()
            logger.debug(f"Database session closed after length calculation.")

    return total_length

def calculate_remaining_length(prior_version: int, working_version: int) -> Optional[int]:
    """
    Calculates the total core_text length of sections from the prior_version
    that have NOT yet been processed in the working_version.

    Args:
        prior_version: The version number of the last complete state (0 for original).
        working_version: The current version number being processed.

    Returns:
        The total character length of unprocessed text, or None on error.
    """
    if prior_version < 0 or working_version <= prior_version:
        logger.error(f"Invalid version numbers: prior={prior_version}, working={working_version}. Working must be > prior.")
        return None

    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        return None

    remaining_length: Optional[int] = None
    try:
        # 1. Get the IDs of sections already processed in the working_version
        processed_ids_query = select(UsCodeSectionRevised.orig_section_id)\
                              .where(UsCodeSectionRevised.version == working_version)
        # Use subquery() to use this in the NOT IN clause effectively
        processed_ids_subquery = processed_ids_query.subquery()
        logger.info(f"Identifying sections already processed in version {working_version}...")

        # 2. Query the source version (prior_version) for the length of remaining sections
        if prior_version == 0:
            logger.info("Calculating remaining length based on original sections (version 0)...")
            source_table = UsCodeSection
            id_column = UsCodeSection.id
            text_column = UsCodeSection.core_text

            remaining_length_query = db.query(
                sql_func.sum(sql_func.length(sql_func.coalesce(text_column, '')))
            ).filter(
                id_column.notin_(select(processed_ids_subquery.c.orig_section_id)) # Filter out processed IDs
            )

        else: # prior_version > 0
            logger.info(f"Calculating remaining length based on revised version {prior_version}...")
            source_table = UsCodeSectionRevised
            id_column = UsCodeSectionRevised.orig_section_id # Use original ID for consistency
            text_column = UsCodeSectionRevised.core_text

            remaining_length_query = db.query(
                sql_func.sum(sql_func.length(sql_func.coalesce(text_column, '')))
            ).filter(
                source_table.version == prior_version, # Only consider the prior version
                id_column.notin_(select(processed_ids_subquery.c.orig_section_id)) # Filter out processed IDs
            )

        remaining_length = remaining_length_query.scalar()

        # Ensure result is an integer, default to 0 if None
        remaining_length = int(remaining_length) if remaining_length is not None else 0
        logger.info(f"Calculated remaining length to process for working version {working_version}: {remaining_length:,}")

    except Exception as e:
        logger.error(f"Error calculating remaining text length (prior={prior_version}, working={working_version}): {e}", exc_info=True)
        remaining_length = None
    finally:
        if db:
            db.close()
            logger.debug("Database session closed after remaining length calculation.")

    return remaining_length

def calculate_revenue_deviation(working_version: int) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
    """
    Calculates the deviation of the current working version's estimated revenue
    from the target revenue (total original impact).

    Args:
        working_version: The current version number being processed (must be >= 1).

    Returns:
        A tuple containing (deviation, current_estimated_revenue, target_revenue)
        as Decimals, or None if an error occurs or working_version is invalid.
        Deviation = Current Estimate - Target Revenue.
    """
    if working_version < 1:
        logger.error("Working version must be 1 or greater to calculate deviation.")
        return None

    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        return None

    deviation: Optional[Decimal] = None
    current_estimated_revenue: Optional[Decimal] = None
    target_revenue: Optional[Decimal] = None

    try:
        # 1. Calculate Target Revenue (Sum of all original SectionImpact)
        target_revenue_query = db.query(
            sql_func.sum(sql_func.coalesce(SectionImpact.revenue_impact, Decimal(0)))
        )
        target_revenue = target_revenue_query.scalar()
        target_revenue = Decimal(target_revenue) if target_revenue is not None else Decimal(0)
        logger.info(f"Target Revenue (Total Original Impact): {target_revenue:,.2f}")

        # 2. Get IDs processed in the working version
        processed_ids_query = select(UsCodeSectionRevised.orig_section_id)\
                              .where(UsCodeSectionRevised.version == working_version)
        processed_ids_subquery = processed_ids_query.subquery()
        logger.info(f"Identifying sections processed in version {working_version}...")

        # 3. Calculate Revenue from Processed Sections (Working Version)
        processed_revenue_query = db.query(
            sql_func.sum(sql_func.coalesce(UsCodeSectionRevised.revised_financial_impact, Decimal(0)))
        ).filter(UsCodeSectionRevised.version == working_version)
        processed_revenue = processed_revenue_query.scalar()
        processed_revenue = Decimal(processed_revenue) if processed_revenue is not None else Decimal(0)
        logger.info(f"Revenue from processed sections (Version {working_version}): {processed_revenue:,.2f}")

        # 4. Calculate Revenue from Remaining Unprocessed Sections (Original Impact)
        remaining_revenue_query = db.query(
            sql_func.sum(sql_func.coalesce(SectionImpact.revenue_impact, Decimal(0)))
        ).join(
             UsCodeSection, SectionImpact.section_id == UsCodeSection.id # Join needed for ID check
        ).filter(
             UsCodeSection.id.notin_(select(processed_ids_subquery.c.orig_section_id)) # Use the NOT IN subquery
        )
        remaining_revenue = remaining_revenue_query.scalar()
        remaining_revenue = Decimal(remaining_revenue) if remaining_revenue is not None else Decimal(0)
        logger.info(f"Revenue from remaining unprocessed sections (Original Impact): {remaining_revenue:,.2f}")

        # 5. Calculate Total Current Estimated Revenue
        current_estimated_revenue = processed_revenue + remaining_revenue
        logger.info(f"Current Estimated Total Revenue (Version {working_version} Processed + Remaining Original): {current_estimated_revenue:,.2f}")

        # 6. Calculate Deviation
        deviation = current_estimated_revenue - target_revenue
        logger.info(f"Calculated Revenue Deviation: {deviation:,.2f}")

    except Exception as e:
        logger.error(f"Error calculating revenue deviation for working version {working_version}: {e}", exc_info=True)
        # Reset values on error
        deviation, current_estimated_revenue, target_revenue = None, None, None
    finally:
        if db:
            db.close()
            logger.debug("Database session closed after deviation calculation.")

    # Return None if any calculation failed
    if deviation is None or current_estimated_revenue is None or target_revenue is None:
         return None
    else:
         return deviation, current_estimated_revenue, target_revenue

# Example usage block for testing
if __name__ == "__main__":
    print("Testing version determination...")
    p_version = 0 # Default values in case determination fails
    w_version = 1
    try:
        p_version, w_version = determine_version_numbers()
        print(f"Result -> Prior Version: {p_version}, Working Version: {w_version}")
    except Exception as e:
        print(f"Version determination test failed with error: {e}")

    print("\nTesting text length calculation...")
    # Test prior version (might be 0 or higher)
    try:
        length_prior = get_total_text_length_for_version(p_version)
        if length_prior is not None:
            print(f"Result -> Total Length for Prior Version ({p_version}): {length_prior:,}")
        else:
            print(f"Failed to get length for prior version ({p_version}).")
    except Exception as e:
        print(f"Test for prior version ({p_version}) failed with error: {e}")

    # Test working version (might be 1 or higher)
    try:
        # Note: Length for working_version will often be 0 until the run completes
        length_working = get_total_text_length_for_version(w_version)
        if length_working is not None:
            print(f"Result -> Total Length for Working Version ({w_version}): {length_working:,}")
        else:
            print(f"Failed to get length for working version ({w_version}).")
    except Exception as e:
        print(f"Test for working version ({w_version}) failed with error: {e}")

    print("\nTesting remaining length calculation...")
    try:
        length_remaining = calculate_remaining_length(p_version, w_version)
        if length_remaining is not None:
             print(f"Result -> Remaining Length for Working Version ({w_version}) based on Prior ({p_version}): {length_remaining:,}")
        else:
            print(f"Failed to calculate remaining length for working version {w_version}.")
    except Exception as e:
        print(f"Test for remaining length (prior={p_version}, working={w_version}) failed with error: {e}")

    print("\nTesting revenue deviation calculation...")
    try:
        # Calculate deviation for the determined working version
        deviation_result = calculate_revenue_deviation(w_version)
        if deviation_result:
             dev, current, target = deviation_result
             print(f"Result -> Target Revenue : {target:,.2f}")
             print(f"Result -> Current Est. Revenue (v{w_version}): {current:,.2f}")
             print(f"Result -> Revenue Deviation: {dev:,.2f}")
        elif w_version >= 1: # Only print failure if expected to run
            print(f"Failed to calculate revenue deviation for working version {w_version}.")
        else:
             print(f"Skipping deviation calculation (working version {w_version} < 1).") # Should not happen with determine_version_numbers logic
    except Exception as e:
        print(f"Test for revenue deviation (working={w_version}) failed with error: {e}") 