# ai_tax_agent/parsers/json_utils.py

import os
import glob
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from pydantic import BaseModel, Field, field_validator

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

# --- Pydantic Model for Combined Output --- #

class CombinedFormLineItem(BaseModel):
    """Represents a single line item aggregated across different amount units."""
    form_title: str = Field(description="The main title of the form (e.g., Form 1040).")
    schedule_title: Optional[str] = Field(default=None, description="The specific schedule title, if applicable (e.g., Schedule A).")
    line_item_number: str = Field(description="The number designation of the line item (e.g., '1a', '7').")
    label: str = Field(description="The descriptive label text for the line item.")
    dollars: Optional[float] = Field(default=None, description="The aggregated amount in dollars.")
    individuals: Optional[float] = Field(default=None, description="The aggregated count of individuals/returns.")
    forms: Optional[float] = Field(default=None, description="The aggregated count of forms.")

    @field_validator('dollars', 'individuals', 'forms', mode='before')
    @classmethod
    def clean_numeric_strings(cls, v):
        """Attempt to clean common numeric string issues before float conversion."""
        if isinstance(v, str):
            # Remove commas
            v = v.replace(",", "")
            # Handle parentheses for negatives
            if v.startswith('(') and v.endswith(')'):
                v = '-' + v[1:-1]
            # Handle potential non-breaking spaces or other whitespace
            v = v.strip()
        # Let Pydantic handle the final float conversion / validation
        return v

# --- Helper Functions --- #

def _parse_amount(amount_str: Optional[str]) -> Optional[float]:
    """Safely parses a string amount into a float, handling None, commas, parentheses."""
    if amount_str is None:
        return None
    if not isinstance(amount_str, str):
         # If it's already a number (int/float), return as float
        if isinstance(amount_str, (int, float)):
             return float(amount_str)
        # Otherwise, log warning and return None
        logger.warning(f"Expected string amount, got {type(amount_str)}: {amount_str}. Returning None.")
        return None
        
    try:
        # Remove commas
        cleaned_str = amount_str.replace(",", "")
        # Handle parentheses for negatives
        if cleaned_str.startswith('(') and cleaned_str.endswith(')'):
            cleaned_str = '-' + cleaned_str[1:-1]
        # Attempt conversion
        return float(cleaned_str)
    except (ValueError, TypeError):
        logger.debug(f"Could not parse amount string '{amount_str}' to float.")
        return None

# --- Main Parsing Function --- #

def parse_tax_stats_json(directory_path: str) -> List[CombinedFormLineItem]:
    """Parses all *.json files in a directory, combines line items by unique key, 
       and aggregates amounts based on the file's inferred unit.

    Args:
        directory_path: The path to the directory containing the JSON files 
                        (e.g., 'data/tax_statistics').

    Returns:
        A list of CombinedFormLineItem objects.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return []

    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in directory: {directory_path}")
        return []

    # Dictionary to aggregate data: 
    # Key: (form_title, schedule_title, line_item_number, label)
    # Value: Dict containing identifying info and amounts per unit
    aggregated_data: Dict[Tuple[str, Optional[str], str, str], Dict[str, Any]] = defaultdict(lambda: {
        'form_title': None,
        'schedule_title': None,
        'line_item_number': None,
        'label': None,
        'dollars': None,
        'individuals': None,
        'forms': None
    })

    logger.info(f"Processing {len(json_files)} JSON files from {directory_path}...")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        logger.debug(f"Processing file: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}")
            continue
        except IOError as e:
            logger.error(f"Error reading file {filename}: {e}")
            continue

        if not isinstance(data, list):
            logger.warning(f"Skipping {filename}: Expected a list of objects, got {type(data)}.")
            continue

        file_amount_unit: Optional[AmountUnit] = None
        processed_items_in_file = 0

        for page_data in data:
            if not isinstance(page_data, dict):
                logger.warning(f"Skipping non-dict item in {filename}: {str(page_data)[:100]}")
                continue

            # Determine amount unit for this file (use first valid entry)
            if file_amount_unit is None:
                unit_str = page_data.get('amount_unit')
                if unit_str:
                    try:
                        file_amount_unit = AmountUnit(unit_str)
                        logger.info(f"Determined unit for {filename} as: {file_amount_unit.name}")
                    except ValueError:
                        logger.warning(f"Unknown amount_unit '{unit_str}' in {filename}. Treating as UNKNOWN.")
                        file_amount_unit = AmountUnit.UNKNOWN
            
            current_amount_unit = file_amount_unit or AmountUnit.UNKNOWN
            unit_key = current_amount_unit.value # e.g., 'dollars', 'individuals', 'forms'
            if unit_key == 'unknown':
                 logger.debug(f"Skipping items in {filename} for page/form '{page_data.get('form_title')}' due to unknown unit.")
                 continue # Skip if unit cannot be determined for aggregation

            form_title = page_data.get('form_title')
            schedule_title = page_data.get('schedule_title') # Can be None
            line_items = page_data.get('line_items')

            if not form_title or not isinstance(line_items, list):
                logger.debug(f"Skipping entry in {filename}: Missing form_title or invalid line_items.")
                continue

            for item in line_items:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict line item in {filename} for form '{form_title}': {str(item)[:100]}")
                    continue
                
                line_num = item.get('line_item_number')
                label = item.get('label')
                amount_str = item.get('amount') # Amount specific to this file's unit

                if not line_num or label is None: # Label can be empty string, but not None
                    logger.debug(f"Skipping line item in {filename} for form '{form_title}': Missing number or label.")
                    continue

                # Ensure line_num is string
                line_num = str(line_num)
                
                # Create unique key
                item_key = (form_title, schedule_title, line_num, label)
                
                # Get or initialize aggregated record
                record = aggregated_data[item_key]
                
                # Populate identifying info if first time seeing this key
                if record['form_title'] is None:
                    record['form_title'] = form_title
                    record['schedule_title'] = schedule_title
                    record['line_item_number'] = line_num
                    record['label'] = label
                
                # Parse and store the amount for the current unit
                # Only update if not already set or if new value is not None
                parsed_amount = _parse_amount(amount_str)
                if parsed_amount is not None and record[unit_key] is None: 
                    record[unit_key] = parsed_amount
                elif parsed_amount is not None and record[unit_key] is not None:
                     # Handle potential conflicts if the same key appears multiple times within the *same unit file*
                     # This shouldn't happen with the current JSON structure, but good to be aware of.
                     # For now, we overwrite, assuming the last one seen is correct, or log a warning.
                     logger.warning(f"Duplicate key {item_key} found within file {filename} (unit: {unit_key}). Overwriting amount {record[unit_key]} with {parsed_amount}.")
                     record[unit_key] = parsed_amount
                
                processed_items_in_file += 1

        logger.debug(f"Processed {processed_items_in_file} items from {filename}.")

    # Convert aggregated data to list of Pydantic objects
    combined_list: List[CombinedFormLineItem] = []
    logger.info(f"Converting {len(aggregated_data)} aggregated records to Pydantic objects...")
    for record_data in aggregated_data.values():
        try:
            combined_item = CombinedFormLineItem(**record_data)
            combined_list.append(combined_item)
        except Exception as e:
            logger.error(f"Error creating Pydantic object for record: {record_data}. Error: {e}")
            continue # Skip invalid records
            
    logger.info(f"Successfully created {len(combined_list)} combined line item objects.")
    return combined_list

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