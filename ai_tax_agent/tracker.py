import logging
from typing import Optional, Dict, Any
from decimal import Decimal # Import Decimal

# Import functions from the versioning module
try:
    from ai_tax_agent.database.versioning import (
        determine_version_numbers,
        get_total_text_length_for_version,
        calculate_remaining_length,
        calculate_revenue_deviation
    )
    # Need SectionImpact and get_session for the fallback target_dollars calculation
    from ai_tax_agent.database.session import get_session 
    from ai_tax_agent.database.models import SectionImpact
    from sqlalchemy import func as sql_func
except ImportError:
    # Handle cases where script might be run directly and path isn't set up
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai_tax_agent.database.versioning import (
        determine_version_numbers,
        get_total_text_length_for_version,
        calculate_remaining_length,
        calculate_revenue_deviation
    )
    from ai_tax_agent.database.session import get_session 
    from ai_tax_agent.database.models import SectionImpact
    from sqlalchemy import func as sql_func

logger = logging.getLogger(__name__)

class TaxSimplificationState:
    """
    Represents the current state of the tax simplification process,
    derived from database values.
    """
    def __init__(self, length_reduction_target: float = 0.5):
        """
        Initializes the state by querying the database.

        Args:
            length_reduction_target: The desired fractional reduction
                                     in total code length (e.g., 0.5 for 50%).
        """
        if not 0.0 <= length_reduction_target < 1.0:
             raise ValueError("length_reduction_target must be between 0.0 (inclusive) and 1.0 (exclusive)")

        self.length_reduction_target: float = length_reduction_target

        # Attributes to be populated from DB
        self.prior_version: Optional[int] = None
        self.working_version: Optional[int] = None
        self.original_character_length: Optional[int] = None
        self.prior_character_length: Optional[int] = None
        self.current_character_length: Optional[int] = None
        self.target_character_length: Optional[int] = None
        self.remaining_length_to_process: Optional[int] = None
        self.target_dollars: Optional[Decimal] = None
        self.current_estimated_revenue: Optional[Decimal] = None
        self.revenue_deviation: Optional[Decimal] = None
        self.load_error: Optional[str] = None # Store error message if loading fails

        self._load_state_from_db()

    def _load_state_from_db(self):
        """Queries the database using versioning functions to populate state."""
        logger.info("Loading tax simplification state from database...")
        try:
            # 1. Determine Versions
            self.prior_version, self.working_version = determine_version_numbers()
            logger.debug(f"Loaded versions: Prior={self.prior_version}, Working={self.working_version}")

            # 2. Get Original Length (Basis for Target Length)
            self.original_character_length = get_total_text_length_for_version(0)
            if self.original_character_length is None:
                # This is critical, cannot proceed without original length
                raise ValueError("Failed to get original text length (version 0) from database.")
            logger.debug(f"Loaded original length (v0): {self.original_character_length}")

            # 3. Calculate Target Length
            self.target_character_length = int(self.original_character_length * (1.0 - self.length_reduction_target))
            logger.debug(f"Calculated target length: {self.target_character_length}")

            # 4. Get Prior and Current Lengths
            self.prior_character_length = get_total_text_length_for_version(self.prior_version)
            self.current_character_length = get_total_text_length_for_version(self.working_version)
            logger.debug(f"Loaded prior length (v{self.prior_version}): {self.prior_character_length}")
            logger.debug(f"Loaded current length (v{self.working_version}): {self.current_character_length}")

            # 5. Calculate Remaining Length
            self.remaining_length_to_process = calculate_remaining_length(self.prior_version, self.working_version)
            logger.debug(f"Calculated remaining length: {self.remaining_length_to_process}")

            # 6. Calculate Revenue Deviation
            # Only calculate deviation if we have a working version > 0
            if self.working_version >= 1:
                deviation_result = calculate_revenue_deviation(self.working_version)
                if deviation_result:
                    self.revenue_deviation, self.current_estimated_revenue, self.target_dollars = deviation_result
                    logger.debug(f"Loaded revenue: Target={self.target_dollars}, Current={self.current_estimated_revenue}, Deviation={self.revenue_deviation}")
                else:
                    logger.warning(f"Could not calculate revenue deviation for working version {self.working_version}.")
                    # Set revenue fields to None if calculation failed
                    self.revenue_deviation, self.current_estimated_revenue, self.target_dollars = None, None, None
            else:
                 # If working_version is 0 (shouldn't happen with current logic, but defensive)
                 # or if deviation isn't applicable yet. Get target dollars separately.
                 logger.info("Working version is < 1 or deviation not calculated, fetching target dollars only.")
                 # Fetch target dollars directly if needed (slightly redundant query)
                 db_temp = None
                 try:
                     db_temp = get_session()
                     if db_temp:
                         target_q = db_temp.query(sql_func.sum(sql_func.coalesce(SectionImpact.revenue_impact, Decimal(0))))
                         self.target_dollars = Decimal(target_q.scalar() or 0)
                         logger.debug(f"Fetched target dollars separately: {self.target_dollars}")
                     else:
                          logger.error("Failed to get temporary session for target dollar fallback.")
                          self.target_dollars = None
                 except Exception as rev_e:
                      logger.error(f"Failed to get target dollars separately: {rev_e}")
                      self.target_dollars = None
                 finally:
                      if db_temp:
                           db_temp.close()
                 self.revenue_deviation = None
                 self.current_estimated_revenue = None

            logger.info("Successfully loaded tax simplification state.")

        except Exception as e:
            logger.error(f"Failed to load simplification state from database: {e}", exc_info=True)
            self.load_error = str(e)
            # Reset key attributes to None on load failure
            self.prior_version = None
            self.working_version = None
            # ... (reset others as needed)
            self.original_character_length = None
            self.prior_character_length = None
            self.current_character_length = None
            self.target_character_length = None
            self.remaining_length_to_process = None
            self.target_dollars = None
            self.current_estimated_revenue = None
            self.revenue_deviation = None

    def summary(self) -> Dict[str, Any]:
        """
        Returns a descriptive summary of the current simplification state
        suitable for use as LLM context.
        """
        if self.load_error:
             # Provide a clear error structure
             return {
                 "status": "Error",
                 "error_message": f"Failed to load state: {self.load_error}"
             }

        # Format values, handling potential None types gracefully
        original_length_str = f"{self.original_character_length:,}" if self.original_character_length is not None else "N/A"
        prior_length_str = f"{self.prior_character_length:,}" if self.prior_character_length is not None else "N/A"
        current_length_str = f"{self.current_character_length:,}" if self.current_character_length is not None else "N/A"
        target_length_str = f"{self.target_character_length:,}" if self.target_character_length is not None else "N/A"
        remaining_length_str = f"{self.remaining_length_to_process:,}" if self.remaining_length_to_process is not None else "N/A"
        target_dollars_str = f"${float(self.target_dollars):,.2f}" if self.target_dollars is not None else "N/A"
        current_revenue_str = f"${float(self.current_estimated_revenue):,.2f}" if self.current_estimated_revenue is not None else "N/A"
        deviation_str = f"${float(self.revenue_deviation):,.2f}" if self.revenue_deviation is not None else "N/A"


        return {
            "status": "Success",
            "prior_completed_version": self.prior_version,
            "current_working_version": self.working_version,
            "original_total_text_length": self.original_character_length,
            "original_total_text_length_description": f"Initial character count before simplification (Version 0): {original_length_str}",
            "prior_version_total_text_length": self.prior_character_length,
            "prior_version_total_text_length_description": f"Total character count of the last completed version ({self.prior_version}): {prior_length_str}",
            "current_version_processed_text_length": self.current_character_length,
            "current_version_processed_text_length_description": f"Character count processed so far in the current working version ({self.working_version}): {current_length_str}",
            "target_character_length": self.target_character_length,
            "target_character_length_description": f"Calculated target character count for the final simplified code ({100*(1.0-self.length_reduction_target):.0f}% of original): {target_length_str}",
            "remaining_text_length_to_process": self.remaining_length_to_process,
            "remaining_text_length_to_process_description": f"Estimated characters remaining to process in the current working version ({self.working_version}): {remaining_length_str}",
            "target_total_revenue": self.target_dollars,
            "target_total_revenue_description": f"Baseline total revenue impact from original code: {target_dollars_str}",
            "current_estimated_total_revenue": self.current_estimated_revenue,
            "current_estimated_total_revenue_description": f"Estimated total revenue impact based on current working version ({self.working_version}): {current_revenue_str}",
            "current_revenue_deviation_from_target": self.revenue_deviation,
            "current_revenue_deviation_from_target_description": f"Difference between current estimated revenue and target revenue: {deviation_str}",
            "length_reduction_target_fraction": self.length_reduction_target
        }

# Removed prior methods: clear, increment_current_length, set_prior_state,
# decrement_remaining_length, track_kept_dollars, track_removed_dollars

# Example usage block for testing
if __name__ == "__main__":
    import json # For pretty printing the summary
    from decimal import Decimal # Need Decimal in this scope for the helper

    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')

    # Helper function to convert Decimal to float for JSON serialization
    def decimal_default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    print("--- Testing TaxSimplificationState Initialization ---")

    try:
        # Initialize the state (using default 0.5 reduction target)
        state = TaxSimplificationState(length_reduction_target=0.5)

        # Get the summary dictionary
        state_summary = state.summary()

        # Print the pretty-printed JSON summary
        print("\n--- State Summary ---")
        print(json.dumps(state_summary, indent=2, default=decimal_default))

        # Optionally print specific values if loading was successful
        if state_summary.get("status") == "Success":
             print(f"\nWorking Version: {state.working_version}")
             print(f"Prior Version: {state.prior_version}")
             print(f"Target Character Length: {state.target_character_length}")
             print(f"Revenue Deviation: {state.revenue_deviation}")
        elif state_summary.get("status") == "Error":
             print(f"\nError loading state: {state_summary.get('error_message')}")

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
        print(f"\nAn error occurred during testing: {e}")

    print("\n--- Test Complete ---")

