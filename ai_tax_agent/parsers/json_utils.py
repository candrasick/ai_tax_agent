# ai_tax_agent/parsers/json_utils.py

import os
import glob
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re

# Assuming AmountUnit is defined elsewhere, e.g., in pdf_parser_utils
# If not, define it here or import appropriately.
# For now, let's try importing it:
try:
    from .pdf_parser_utils import AmountUnit
except ImportError:
    # Fallback if structure doesn't allow direct import, define locally
    from enum import Enum
    class AmountUnit(Enum):
        DOLLARS = "dollars"
        INDIVIDUALS = "individuals"
        FORMS = "forms"
        UNKNOWN = "unknown"

logger = logging.getLogger(__name__)

# Define the TaxType enum
class TaxType(str, Enum):
    INDIVIDUALS = "individuals"
    PARTNERSHIPS = "partnerships"
    CORPORATIONS = "corporations"
    UNKNOWN = "unknown" # Add a fallback

# New Pydantic model for individual line items
class TaxStatsLineItem(BaseModel):
    form_title: str
    schedule_title: Optional[str] = None
    line_item_number: str
    label: str
    amount_unit: str # e.g., "dollars", "individuals", "forms"
    amount: float
    tax_type: TaxType
    full_text: str

    model_config = {
        "use_enum_values": True  # Serialize enum values as strings
    }

# --- Helper Functions --- #

def _parse_amount(amount_str: Optional[str]) -> Optional[float]:
    """Safely parses a string amount, removing commas and handling parentheses for negatives."""
    if amount_str is None or not isinstance(amount_str, str):
        return None
    
    # Remove commas
    amount_str = amount_str.replace(',', '')
    
    # Check for parentheses indicating negative numbers (e.g., "(1234)")
    is_negative = False
    if amount_str.startswith('(') and amount_str.endswith(')'):
        is_negative = True
        amount_str = amount_str[1:-1] # Remove parentheses
        
    # Use regex to find the first valid number pattern (integer or decimal)
    match = re.search(r'^-?\d+(\.\d+)?', amount_str)
    if match:
        try:
            amount = float(match.group(0))
            return -amount if is_negative else amount
        except ValueError:
            # Log if conversion fails after regex match (should be rare)
            logger.warning(f"Could not convert matched numeric string to float: '{match.group(0)}' from original '{amount_str}'")
            return None
    else:
        # Log if no numeric pattern is found
        # logger.debug(f"No valid numeric pattern found in amount string: '{amount_str}'") # Can be noisy
        return None

def _determine_tax_type(filename: str) -> TaxType:
    """Determines the TaxType based on the filename."""
    base_name = os.path.basename(filename).lower()
    if base_name.startswith("individuals"):
        return TaxType.INDIVIDUALS
    elif base_name.startswith("partnerships"):
        return TaxType.PARTNERSHIPS
    elif base_name.startswith("corporations"):
        return TaxType.CORPORATIONS
    else:
        logger.warning(f"Could not determine TaxType from filename: {filename}. Defaulting to UNKNOWN.")
        return TaxType.UNKNOWN

def _generate_full_text(item: 'TaxStatsLineItem') -> str:
    """Generates the descriptive full_text string."""
    schedule_part = f" and schedule {item.schedule_title}" if item.schedule_title else ""
    # Ensure amount is formatted reasonably, e.g., with commas for readability
    formatted_amount = f"{item.amount:,.2f}" if isinstance(item.amount, (int, float)) else str(item.amount)
    
    # Defensively handle tax_type being either enum member or string value
    if isinstance(item.tax_type, Enum):
        tax_type_str = item.tax_type.value
    else:
        tax_type_str = str(item.tax_type) # Assume it's already the string value

    return (
        f"For {tax_type_str}, in form {item.form_title}{schedule_part}, "
        f"{item.line_item_number}.{item.label} has {formatted_amount} {item.amount_unit}."
    )

# --- Main Parsing Function --- #

def parse_tax_stats_json(directory: str) -> List[TaxStatsLineItem]:
    """
    Parses all JSON files in a directory, creating an individual record for each 
    amount found, determines TaxType from filename, and generates descriptive text.

    Args:
        directory: The path to the directory containing the tax statistics JSON files.

    Returns:
        A list of TaxStatsLineItem objects, each representing a single data point.
    """
    all_line_items: List[TaxStatsLineItem] = []
    json_files = glob.glob(os.path.join(directory, '*.json'))

    if not json_files:
        logger.warning(f"No JSON files found in directory: {directory}")
        return []

    logger.info(f"Found {len(json_files)} JSON files to process in {directory}")

    for file_path in json_files:
        logger.debug(f"Processing file: {file_path}")
        tax_type = _determine_tax_type(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from file: {file_path}", exc_info=True)
            continue
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            continue

        # --- Adapt based on actual JSON structure ---
        # Assuming the structure is a list of objects, each with form/schedule/line/label
        # and potentially nested amounts like {"dollars": "1,234", "individuals": "56"}
        # We need to iterate and extract each amount unit separately.
        
        if not isinstance(data, list):
             logger.warning(f"Expected a list of items in {file_path}, but got {type(data)}. Skipping file.")
             continue

        for page_data in data: # Rename item_data to page_data for clarity
            if not isinstance(page_data, dict):
                logger.warning(f"Expected page/section data to be a dict in {file_path}, but got {type(page_data)}. Skipping.")
                continue

            # DEBUG: Log the keys found in the current page/section item
            # logger.debug(f"Processing page/section with keys: {list(page_data.keys())}") # Removed/Commented out

            # Extract form/schedule info and amount unit from the outer dictionary
            form_title = page_data.get("form_title")
            schedule_title = page_data.get("schedule_title") # May be None
            page_amount_unit_str = page_data.get("amount_unit") # Get the unit string for this page/section

            # Validate the amount unit string
            if not page_amount_unit_str or page_amount_unit_str not in ["dollars", "individuals", "forms", "returns"]:
                logger.warning(f"Skipping page/section in {file_path} due to missing or invalid 'amount_unit': {page_amount_unit_str}")
                continue
            
            # Get the list of actual line items
            line_items_list = page_data.get("line_items")

            # Check if essential fields exist at the page level and if line_items is a list
            if not form_title or not isinstance(line_items_list, list):
                 # logger.debug(f"Skipping page/section due to missing 'form_title' or invalid 'line_items' (type: {type(line_items_list)}). Data: {page_data}") # Removed/Commented out
                 continue
            
            # logger.debug(f"Found {len(line_items_list)} line items within this page/section (unit: {page_amount_unit_str}).") # Removed/Commented out

            # --- Inner loop for actual line items ---            
            for line_item_data in line_items_list:
                if not isinstance(line_item_data, dict):
                    logger.warning(f"Expected line item to be a dict in {file_path} (form: {form_title}), but got {type(line_item_data)}. Skipping line item.")
                    continue
                
                # logger.debug(f"  Processing line item with keys: {list(line_item_data.keys())}") # Removed/Commented out

                # Extract details from the inner line_item_data dictionary
                base_info = {
                    "form_title": form_title, # From outer dict
                    "schedule_title": schedule_title, # From outer dict
                    "line_item_number": line_item_data.get("line_item_number"),
                    "label": line_item_data.get("label"),
                    "tax_type": tax_type, # From file
                }

                # Check if all required base fields for the *line item* are present
                required_fields_present = all([base_info["line_item_number"], base_info["label"] is not None]) # form_title is checked above
                if not required_fields_present:
                     # logger.debug(f"    Skipping line item due to missing number or label. Found: number={base_info['line_item_number']}, label_present={base_info['label'] is not None}") # Removed/Commented out
                     continue

                # Get the raw amount directly from the 'amount' key
                raw_amount = line_item_data.get("amount") 
                # logger.debug(f"    Checking 'amount' key: raw_amount = {repr(raw_amount)} (type: {type(raw_amount)}) within line item") # Removed/Commented out
                
                if raw_amount is not None:
                    parsed_amount = _parse_amount(str(raw_amount))
                    # logger.debug(f"      Parsed amount: {parsed_amount}") # Removed/Commented out
                    
                    if parsed_amount is not None:
                        try:
                            line_item = TaxStatsLineItem(
                                **base_info, # type: ignore
                                amount_unit=page_amount_unit_str, # Use unit from page_data
                                amount=parsed_amount,
                                full_text="" # Placeholder
                            )
                            line_item.full_text = _generate_full_text(line_item)
                            all_line_items.append(line_item)
                            # logger.debug(f"      Successfully created TaxStatsLineItem for unit '{page_amount_unit_str}'") # Removed/Commented out
                        except Exception as e:
                            logger.error(f"Failed to create TaxStatsLineItem for unit '{page_amount_unit_str}' in {file_path} line item: {line_item_data}. Error: {e}", exc_info=True)
                    # else: # Removed/Commented out else block for missing parse
                         # logger.debug(f"    Raw amount '{repr(raw_amount)}' could not be parsed to a float.") 
                # else: # Removed/Commented out else block for missing amount key
                     # logger.debug(f"    'amount' key was missing or None in line item: {line_item_data}")

            # --- End inner loop ---

    logger.info(f"Finished processing. Generated {len(all_line_items)} individual line item records.")
    return all_line_items

# Example Usage (can be removed or placed in a separate script):
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     # Assuming your script is run from the project root or ai_tax_agent directory
#     stats_dir = os.path.join('..', 'data', 'tax_statistics') # Adjust path if needed
#     if not os.path.exists(stats_dir):
#          stats_dir = os.path.join('.', 'data', 'tax_statistics') # Try relative from current dir

#     if os.path.exists(stats_dir):
#         combined_data = parse_tax_stats_json(stats_dir)
#         print(f"\n--- Parsed {len(combined_data)} Combined Line Items ---")
#         # Print first 5 examples
#         for i, item in enumerate(combined_data[:5]):
#             print(f"\nItem {i+1}:")
#             print(item.model_dump_json(indent=2))
#     else:
#          print(f"Error: Statistics directory not found at expected location: {stats_dir}") 