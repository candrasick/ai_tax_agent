import logging
import json
from decimal import Decimal

from langchain.tools import Tool

# Assuming TaxSimplificationState is in the tracker module
try:
    from ai_tax_agent.tracker import TaxSimplificationState
except ImportError:
    # Handle potential path issues if run standalone (less likely for tool usage)
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai_tax_agent.tracker import TaxSimplificationState


logger = logging.getLogger(__name__)

# Helper function to handle Decimal serialization for JSON output
def decimal_default_for_tool(obj):
    if isinstance(obj, Decimal):
        # Convert Decimal to string for precision, or float if acceptable
        # Using float here for simplicity as requested in the prompt's example output format
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_current_simplification_state() -> str:
    """
    Retrieves the current overall state of the tax simplification process,
    including version numbers, text lengths (original, current, target),
    and revenue neutrality metrics (target, current estimate, deviation).
    This provides the necessary 'ledger' context for decision-making.
    Returns the state summary as a JSON string.
    """
    logger.info("Retrieving current tax simplification state.")
    try:
        # Instantiate the state tracker - use default reduction target for now,
        # or consider making it configurable if needed later.
        state = TaxSimplificationState()
        state_summary = state.summary()

        # Convert the summary dictionary to a JSON string, handling Decimals
        # Use the helper function to handle potential Decimals in the summary
        json_summary = json.dumps(state_summary, indent=2, default=decimal_default_for_tool)
        logger.debug(f"State summary retrieved: {json_summary}")
        return json_summary

    except Exception as e:
        logger.error(f"Error retrieving simplification state: {e}", exc_info=True)
        # Return an error message as JSON string
        error_state = {
            "status": "Error",
            "error_message": f"Failed to retrieve state: {str(e)}"
        }
        return json.dumps(error_state, indent=2)

# Create the LangChain Tool
get_simplification_state_tool = Tool.from_function(
    func=get_current_simplification_state,
    name="Get Current Simplification State",
    description="Retrieves the current overall project state ('ledger') including progress on text length reduction and revenue neutrality. Use this to understand the context before deciding on an action (simplify, redraft, delete, keep). Returns state as a JSON string.",
    # Optionally add args_schema if you expect specific inputs later
    # args_schema=...
)

# Example usage (for testing the tool function directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("--- Testing get_current_simplification_state ---")
    state_json = get_current_simplification_state()
    print(state_json)
    print("--- Test Complete ---") 