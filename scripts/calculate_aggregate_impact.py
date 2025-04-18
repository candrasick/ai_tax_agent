#!/usr/bin/env python
"""Calculates the total aggregated revenue impact from the section_impact table."""

import logging
import sys
from decimal import Decimal

# Adjust import path based on your project structure
sys.path.append('.') 

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import SectionImpact

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_total_impact() -> Decimal | None:
    """Queries the database and calculates the sum of revenue_impact."""
    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        return None

    total_impact: Decimal | None = None
    try:
        logger.info("Querying database to sum revenue_impact from section_impact table...")
        
        # Query the sum of revenue_impact, treating NULLs as 0
        # coalesce(column, 0) replaces NULL with 0
        total_impact_query = db.query(sql_func.sum(sql_func.coalesce(SectionImpact.revenue_impact, 0)))
        
        # Use .scalar() to get the single sum value
        total_impact = total_impact_query.scalar()
        
        if total_impact is None:
            # This should only happen if the table is empty, sum of empty set might be None depending on DB/SQLAlchemy
            logger.warning("Query returned None for total impact, assuming 0. Table might be empty.")
            total_impact = Decimal(0)
        else:
             # Ensure it's Decimal
             total_impact = Decimal(total_impact)
             logger.info(f"Successfully calculated total impact.")
        # --- REMOVED SCALING --- 
        # if total_impact is not None:
        #     total_impact *= Decimal(10) # Scale the result by 10
        #     logger.info(f"Successfully calculated and scaled total impact by 10.")
        # else:
        #     # Handle case where total_impact remained None (e.g., empty table)
        #      logger.info(f"Total impact is zero or could not be calculated.")
        # -----------------------
        
    except Exception as e:
        logger.error(f"Database error during impact calculation: {e}", exc_info=True)
        total_impact = None # Ensure None is returned on error
    finally:
        if db:
            db.close()
            logger.debug("Database session closed.")
            
    return total_impact

if __name__ == "__main__":
    calculated_total = calculate_total_impact()

    if calculated_total is not None:
        # Format as currency for better readability
        print(f"\nTotal Aggregated Revenue Impact: ${calculated_total:,.2f}")
        logger.info("Calculation complete.")
    else:
        print("\nFailed to calculate total aggregated revenue impact.")
        logger.error("Calculation failed.")
        sys.exit(1) 