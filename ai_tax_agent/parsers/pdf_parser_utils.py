# ai_tax_agent/parsers/pdf_parser_utils.py

import pdfplumber
from typing import List, Tuple, Dict, Any, Optional
import re

def extract_amounts_by_color(
    page: pdfplumber.page.Page,
    amount_color_tuple: Tuple[float, float, float],
    color_tolerance: float = 0.01
) -> List[Dict[str, Any]]:
    """Extracts words from a PDF page suspected to be amounts based on color.

    Args:
        page: The pdfplumber Page object.
        amount_color_tuple: A tuple representing the target RGB color (e.g., (0, 0, 0.5)).
                           Assumes color values are between 0 and 1.
        color_tolerance: Allowable difference for each color component during comparison.

    Returns:
        A list of dictionaries, each representing an amount word found,
        containing 'amount' (text) and 'position' (bounding box tuple).
    """
    amount_list = []
    # Extract words with necessary attributes
    words = page.extract_words(extra_attrs=['fontname', 'size', 'non_stroking_color'])

    for word in words:
        # Check color - handle potential minor float variations
        word_color = word.get('non_stroking_color')
        is_amount_word = False

        # Ensure word_color is a tuple and has at least 3 components (RGB)
        if isinstance(word_color, tuple) and len(word_color) >= 3:
            # Compare RGB values within the specified tolerance
            if (
                abs(word_color[0] - amount_color_tuple[0]) < color_tolerance and
                abs(word_color[1] - amount_color_tuple[1]) < color_tolerance and
                abs(word_color[2] - amount_color_tuple[2]) < color_tolerance
            ):
                is_amount_word = True

        if is_amount_word:
            amount_list.append({
                "amount": word['text'],
                "position": (word['x0'], word['top'], word['x1'], word['bottom'])
            })

    return amount_list

def extract_form_schedule_titles(
    header_elements: List[Dict[str, Any]]
) -> Dict[str, Optional[str]]:
    """Extracts the main Form and Schedule titles from a list of header text elements.

    Args:
        header_elements: A list of dictionaries, where each dictionary represents a
                         text element identified as a header and contains a 'text' key.

    Returns:
        A dictionary with keys 'form_title' and 'schedule_title', containing the
        extracted titles (e.g., "Form 1099", "Schedule K-1") or None if not found.
    """
    form_title = None
    schedule_title = None

    # Regex to find "Form" followed by whitespace and an alphanumeric/hyphenated identifier
    form_pattern = re.compile(r"\bForm\s+([\w-]+)\b", re.IGNORECASE)
    # Regex to find "Schedule" followed by whitespace and an alphanumeric/hyphenated identifier
    schedule_pattern = re.compile(r"\bSchedule\s+([\w-]+)\b", re.IGNORECASE)

    for element in header_elements:
        text = element.get('text', '')
        if not text:
            continue

        # Search for Form title if not already found
        if form_title is None:
            form_match = form_pattern.search(text)
            if form_match:
                # Construct the full title
                form_title = f"Form {form_match.group(1)}"

        # Search for Schedule title if not already found
        if schedule_title is None:
            schedule_match = schedule_pattern.search(text)
            if schedule_match:
                # Construct the full title
                schedule_title = f"Schedule {schedule_match.group(1)}"

        # Optimization: Stop searching if both are found
        if form_title is not None and schedule_title is not None:
            break

    return {"form_title": form_title, "schedule_title": schedule_title} 