# ai_tax_agent/parsers/pdf_parser_utils.py

import pdfplumber
from typing import List, Tuple, Dict, Any, Optional
import re
from enum import Enum
import math # Import math for distance calculation
import os
import logging

# Define Enum for amount units
class AmountUnit(Enum):
    DOLLARS = "dollars"
    INDIVIDUALS = "individuals"
    FORMS = "forms"
    UNKNOWN = "unknown"

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
            # Validate if the text is actually numeric (allowing commas)
            amount_text = word['text']
            text_without_commas = amount_text.replace(",", "")
            is_numeric = False
            try:
                # Attempt to convert to float to handle decimals if any
                float(text_without_commas)
                is_numeric = True
            except ValueError:
                # Not a valid number after removing commas
                is_numeric = False
                # Optional: log skipped non-numeric amounts
                # print(f"Skipping non-numeric amount word: '{amount_text}'")

            # Only append if it passed the numeric check
            if is_numeric:
                amount_list.append({
                    "amount": amount_text, # Store original text with commas
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
        text = element.get('phrase', '')
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

def determine_amount_unit(all_text_phrases: List[Dict[str, Any]]) -> AmountUnit:
    """Determines the unit type (dollars, individuals, forms) based on header text.

    Searches for specific phrases within the text of header elements.

    Args:
        all_text_phrases: A list of dictionaries representing text phrases (headers + body).

    Returns:
        An AmountUnit enum member indicating the determined unit type.
    """
    # Define patterns for case-insensitive matching
    individuals_pattern = re.compile(r"Number\s+of\s+individuals", re.IGNORECASE)
    dollars_pattern = re.compile(r"Amount\s+of\s+selected\s+lines", re.IGNORECASE)
    forms_pattern = re.compile(r"Number\s+of\s+Forms", re.IGNORECASE)

    for element in all_text_phrases:
        text = element.get('phrase', '') # Expecting phrase text now
        if not text:
            continue

        if individuals_pattern.search(text):
            return AmountUnit.INDIVIDUALS
        if dollars_pattern.search(text):
            return AmountUnit.DOLLARS
        if forms_pattern.search(text):
            return AmountUnit.FORMS

    # If none of the patterns are found in any header element
    return AmountUnit.UNKNOWN

def extract_phrases_and_line_items(
    page: pdfplumber.page.Page,
    vertical_tolerance: float = 2,
    horizontal_tolerance: float = 5,
    line_item_font_size_threshold: float = 7.0,
    header_font_size_threshold: float = 7.0, # Reuse or define separately
    page_corner_threshold_pct: float = 0.15
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts line items, header phrases, and body phrases.

    Groups words based on proximity, then identifies and filters line item numbers
    based on pattern, font size, and position.

    Args:
        page: The pdfplumber Page object.
        vertical_tolerance: Max vertical distance between word tops for same phrase.
        horizontal_tolerance: Max horizontal distance between words for same phrase.
        line_item_font_size_threshold: Font size below which a number is considered a line item.
        header_font_size_threshold: Font size threshold for classifying as a header phrase.
        page_corner_threshold_pct: Percentage of page dimensions to define the upper-left corner.

    Returns:
        A dictionary containing three lists:
            'line_item_numbers': Filtered and sorted line item numbers with positions.
            'header_phrases': Filtered header phrases with positions.
            'body_phrases': Remaining text phrases with positions.
    """
    # --- Extract words and sort them --- 
    words = page.extract_words(extra_attrs=['fontname', 'size']) # Need size for filtering
    if not words:
        return {"line_item_numbers": [], "header_phrases": [], "body_phrases": []}
    
    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    # --- Group words into phrases --- 
    phrases_raw = [] # Store phrases as list of words first
    current_phrase_words = []

    for i, word in enumerate(sorted_words):
        if not current_phrase_words:
            current_phrase_words.append(word)
        else:
            last_word = current_phrase_words[-1]
            vertically_aligned = abs(word['top'] - last_word['top']) < vertical_tolerance
            horizontally_close = (word['x0'] > last_word['x1'] - 1) and \
                                 (word['x0'] - last_word['x1']) < horizontal_tolerance

            if vertically_aligned and horizontally_close:
                current_phrase_words.append(word)
            else:
                if current_phrase_words:
                    min_x0 = min(w['x0'] for w in current_phrase_words)
                    min_y0 = min(w['top'] for w in current_phrase_words)
                    max_x1 = max(w['x1'] for w in current_phrase_words)
                    max_y1 = max(w['bottom'] for w in current_phrase_words)
                    phrases_raw.append({
                        "words": list(current_phrase_words),
                        "position": (min_x0, min_y0, max_x1, max_y1)
                    })
                current_phrase_words = [word]

    if current_phrase_words:
        min_x0 = min(w['x0'] for w in current_phrase_words)
        min_y0 = min(w['top'] for w in current_phrase_words)
        max_x1 = max(w['x1'] for w in current_phrase_words)
        max_y1 = max(w['bottom'] for w in current_phrase_words)
        phrases_raw.append({
            "words": list(current_phrase_words),
            "position": (min_x0, min_y0, max_x1, max_y1)
        })

    # --- Separate Line Items, Headers, and Body Phrases ---
    line_item_numbers_unfiltered = []
    header_phrases = []
    body_phrases = []
    line_item_pattern = re.compile(r"^\s*(\d{1,2}[a-zA-Z]?)\s*$")

    for phrase_data in phrases_raw:
        word_list = phrase_data['words']
        is_line_item = False
        first_word_size = word_list[0].get('size') if word_list else None

        # Check for Line Item
        if len(word_list) == 1:
            single_word = word_list[0]
            word_text = single_word['text']
            word_size = single_word.get('size')
            match = line_item_pattern.fullmatch(word_text)
            # Use <= for threshold comparison as per previous logic
            if match and (word_size is None or word_size < line_item_font_size_threshold):
                is_line_item = True
                line_item_numbers_unfiltered.append({
                    "line_item_number": match.group(1),
                    "position": phrase_data['position'],
                    "size": word_size # Keep size info if needed
                })

        # If not a line item, classify as Header or Body
        if not is_line_item:
            phrase_text = " ".join(w['text'] for w in word_list)
            output_phrase = {
                "phrase": phrase_text,
                "position": phrase_data['position']
            }
            # Use >= for header threshold comparison
            if first_word_size is not None and first_word_size >= header_font_size_threshold:
                 header_phrases.append(output_phrase)
            else:
                 body_phrases.append(output_phrase)

    # --- Filter out page numbers from line items ---
    filtered_line_item_numbers = list(line_item_numbers_unfiltered)
    if line_item_numbers_unfiltered:
        most_upper_left_item = min(
            line_item_numbers_unfiltered,
            key=lambda item: (item['position'][1], item['position'][0])
        )
        page_width = page.width
        page_height = page.height
        x_threshold = page_width * page_corner_threshold_pct
        y_threshold = page_height * page_corner_threshold_pct
        pos = most_upper_left_item['position']
        if pos[0] < x_threshold and pos[1] < y_threshold:
            filtered_line_item_numbers.remove(most_upper_left_item)

    # --- Sort line items --- 
    def get_sort_key(item):
        line_num_str = item['line_item_number']
        match = re.match(r"(\d+)([a-zA-Z]?)", line_num_str)
        if match:
            num_part = int(match.group(1))
            letter_part = match.group(2).lower()
            return (num_part, letter_part)
        else:
            return (float('inf'), line_num_str)
    filtered_line_item_numbers.sort(key=get_sort_key)

    return {
        "line_item_numbers": filtered_line_item_numbers,
        "header_phrases": header_phrases,
        "body_phrases": body_phrases
    }

def associate_line_labels(
    line_items: List[Dict[str, Any]],
    text_phrases: List[Dict[str, Any]], # Typically body_phrases
    default_vertical_tolerance: float = 5.0
) -> List[Dict[str, Any]]:
    """Associates line item numbers with the closest, aligned text phrase to their right.

    Args:
        line_items: List of line item dicts (from extract_phrases_and_line_items).
        text_phrases: List of text phrase dicts (from extract_phrases_and_line_items).
        default_vertical_tolerance: Fallback vertical tolerance if item height is small.

    Returns:
        A list of dictionaries, each containing 'line_item_number' and 'label'
        (which is the text of the associated phrase, or None if no match found).
    """
    labeled_lines = []

    for item in line_items:
        item_pos = item['position'] # (x0, top, x1, bottom)
        item_right_x = item_pos[2]
        item_v_center = (item_pos[1] + item_pos[3]) / 2
        item_height = item_pos[3] - item_pos[1]
        # Use half item height for tolerance, with a fallback default
        vertical_tolerance = item_height / 2 if item_height > 1 else default_vertical_tolerance

        best_match_phrase_data = None
        min_horizontal_distance = float('inf')

        for phrase in text_phrases:
            phrase_pos = phrase['position'] # (x0, top, x1, bottom)
            phrase_left_x = phrase_pos[0]
            phrase_v_center = (phrase_pos[1] + phrase_pos[3]) / 2

            if phrase_left_x >= item_right_x and abs(phrase_v_center - item_v_center) < vertical_tolerance:
                horizontal_distance = phrase_left_x - item_right_x
                if horizontal_distance < min_horizontal_distance:
                    min_horizontal_distance = horizontal_distance
                    best_match_phrase_data = phrase

        label_text = best_match_phrase_data['phrase'] if best_match_phrase_data else None
        label_pos = best_match_phrase_data['position'] if best_match_phrase_data else None

        combined_pos = None
        if label_pos:
             min_x0 = item_pos[0]
             min_y0 = min(item_pos[1], label_pos[1])
             max_x1 = label_pos[2]
             max_y1 = max(item_pos[3], label_pos[3])
             combined_pos = (min_x0, min_y0, max_x1, max_y1)

        labeled_lines.append({
            "line_item_number": item['line_item_number'],
            "label": label_text,
            "line_item_position": item_pos, # Keep original positions if needed
            "label_position": label_pos,
            "combined_position": combined_pos # Store combined position
        })
    return labeled_lines

def associate_amounts_to_lines(
    labeled_lines: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    page_height: float, # Needed for distance normalization/thresholding
    max_distance_factor: float = 0.15 # Factor of page height for proximity
) -> Dict[str, str]:
    """Associates amounts with the closest 'up-and-left' labeled line item.

    Args:
        labeled_lines: List of dictionaries containing line item numbers and their labels.
        amounts: List of dictionaries containing amounts and their positions.
        page_height: Height of the PDF page for distance normalization.
        max_distance_factor: Factor of page height for proximity threshold.

    Returns:
        A dictionary mapping line_item_number to amount_text.
    """
    amount_map = {} # Maps line_item_number -> amount_text

    # Filter labeled_lines to only those with a combined_position for association
    valid_lines = [line for line in labeled_lines if line.get("combined_position")]
    # Create a list of amounts not yet assigned
    available_amounts = list(amounts)

    # Iterate through lines, finding the best amount for each
    for line_data in valid_lines:
        line_pos = line_data['combined_position'] # (x0, top, x1, bottom)
        line_v_center = (line_pos[1] + line_pos[3]) / 2

        # --- DEBUG: Use large tolerance initially --- 
        vertical_tolerance = 100.0 # Large fixed tolerance for debugging
        # line_height = line_pos[3] - line_pos[1]
        # vertical_tolerance = max(line_height / 2, 3.0)
        print(f"\nProcessing Line: {line_data['line_item_number']} '{line_data['label'][:30]}...' (v_center: {line_v_center:.2f}, tol: {vertical_tolerance:.2f})")

        best_match_amount_data = None
        min_v_distance = float('inf')

        # Search through amounts that haven't been assigned yet
        for i, amount_data in enumerate(available_amounts):
            amount_pos = amount_data['position'] # (x0, top, x1, bottom)
            amount_v_center = (amount_pos[1] + amount_pos[3]) / 2

            # Check 1: Amount must be to the right of the combined label/number block
            if amount_pos[0] > line_pos[2]:
                # Check 2: Calculate vertical distance
                print(f"  Considering Amount: '{amount_data['amount']}' (v_center: {amount_v_center:.2f}, v_dist: {v_distance:.2f})")
                v_distance = abs(amount_v_center - line_v_center)

                # Check 3: Is it within vertical tolerance and the closest found so far?
                if v_distance < vertical_tolerance and v_distance < min_v_distance:
                    min_v_distance = v_distance
                    best_match_amount_data = amount_data
                    # Store index to remove later if assigning one amount per line
                    # best_match_amount_index = i 
        print(f"  -> Best Match: '{best_match_amount_data['amount'] if best_match_amount_data else 'None'}' (min_v_dist: {min_v_distance if min_v_distance != float('inf') else 'N/A'})" )

        # Assign the best match found for this line
        if best_match_amount_data:
            # Avoid overwriting if multiple amounts map to the same line?
            # Current logic: last amount found wins. Could collect lists instead.
            # Assign amount to the current line item number
            amount_map[line_data['line_item_number']] = best_match_amount_data['amount']
            # Optimization: Remove the assigned amount so it can't be assigned again
            # This assumes one amount per line. Adjust if amounts can align with multiple lines.
            # DEBUG: Temporarily disable removing amounts to see all potential matches
            # try: 
            #     available_amounts.remove(best_match_amount_data)
            # except ValueError: 
            #      pass 

    return amount_map

# --- Main Orchestration Function --- 

def parse_pdf_page_structure(
    pdf_path: str,
    page_num_one_indexed: int,
    amount_color_tuple: Tuple[float, float, float] = (0, 0, 0.5),
    # Pass other parameters with defaults if needed
    vertical_tolerance: float = 2,
    horizontal_tolerance: float = 5,
    line_item_font_size_threshold: float = 7.0,
    header_font_size_threshold: float = 7.0, 
    page_corner_threshold_pct: float = 0.15,
    amount_proximity_factor: float = 0.15

) -> Optional[Dict[str, Any]]:
    """Parses a single PDF page to extract structured line items, labels, and amounts.

    Orchestrates calls to various extraction and association functions.

    Args:
        pdf_path: Path to the PDF file.
        page_num_one_indexed: The 1-based page number to analyze.
        amount_color_tuple: RGB color tuple for identifying amounts.
        vertical_tolerance: Tolerance for phrase grouping.
        horizontal_tolerance: Tolerance for phrase grouping.
        line_item_font_size_threshold: Font size limit for line items.
        header_font_size_threshold: Font size threshold for headers.
        page_corner_threshold_pct: Threshold for filtering page numbers.
        amount_proximity_factor: Proximity factor for associating amounts.

    Returns:
        A dictionary containing the parsed structure ('form_title', 'schedule_title',
        'amount_unit', 'line_items') or None if the page cannot be processed.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return None

    page_index = page_num_one_indexed - 1

    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        with pdfplumber.open(pdf_path) as pdf:
            if page_index < 0 or page_index >= len(pdf.pages):
                logging.error(f"Page number {page_num_one_indexed} is out of range (1-{len(pdf.pages)}).")
                return None

            page = pdf.pages[page_index]

            # --- Extract base elements --- 
            phrase_result = extract_phrases_and_line_items(
                page, vertical_tolerance, horizontal_tolerance,
                line_item_font_size_threshold, header_font_size_threshold,
                page_corner_threshold_pct
            )
            line_items = phrase_result["line_item_numbers"]
            header_phrases = phrase_result["header_phrases"]
            body_phrases = phrase_result["body_phrases"]

            amounts = extract_amounts_by_color(page, amount_color_tuple=amount_color_tuple)

            # --- Determine overall context --- 
            all_phrases_for_unit = header_phrases + body_phrases
            amount_unit = determine_amount_unit(all_phrases_for_unit)
            title_info = extract_form_schedule_titles(header_phrases)

            # --- Associate labels to line items --- 
            labeled_lines = associate_line_labels(line_items, body_phrases)

            # --- Associate amounts to labeled lines --- 
            amount_map = associate_amounts_to_lines(labeled_lines, amounts, page.height, max_distance_factor=amount_proximity_factor)

            # --- Build final output --- 
            output_line_items = []
            unmatched_label_count = 0
            for line_data in labeled_lines:
                if line_data["label"] is None:
                    unmatched_label_count += 1
                    # Optional: Could log here if needed within the utility
                    continue

                line_num = line_data['line_item_number']
                output_line_items.append({
                    "line_item_number": line_num,
                    "label": line_data['label'],
                    "amount": amount_map.get(line_num), 
                })

            # Logging within the utility might be less desirable than in the calling script
            # Consider returning counts or letting the caller handle logging.
            # if unmatched_label_count > 0: ...
            # unmapped_amount_count = len(amounts) - len(...)            
            # if unmapped_amount_count > 0: ...

            final_output = {
                "form_title": title_info["form_title"],
                "schedule_title": title_info["schedule_title"],
                "amount_unit": amount_unit.value,
                "line_items": output_line_items
            }
            return final_output

    except Exception as e:
        logging.error(f"An error occurred during PDF processing for page {page_num_one_indexed}: {e}", exc_info=True)
        return None 