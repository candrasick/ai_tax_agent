# ai_tax_agent/parsers/pdf_parser_utils.py

import pdfplumber
from typing import List, Tuple, Dict, Any, Optional
import re
from enum import Enum
import math # Import math for distance calculation
import os
import logging
import json # Added JSON import
from io import BytesIO # For image conversion
import base64 # For image encoding

# Langchain / LLM imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Assume settings are accessible for API key
from ai_tax_agent.settings import settings

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

# --- Refactored Title Extraction ---

def _attempt_split_title(text: str, schedule_keywords: List[str]) -> Optional[Tuple[str, str]]:
    """Attempts to split text containing both Form and Schedule keywords."""
    for skeyword in schedule_keywords:
        # Split by the schedule keyword, keeping the keyword and everything after it
        parts = re.split(rf"(\b{skeyword}\b.*)", text, 1, re.IGNORECASE)
        if len(parts) >= 3:
            potential_form = parts[0].strip()
            potential_schedule = "".join(parts[1:]).strip()
            # Return only if split seems reasonable (both parts non-empty)
            if potential_form and potential_schedule:
                logging.debug(f"Split title: Form='{potential_form}', Schedule='{potential_schedule}'")
                return potential_form, potential_schedule
    return None # Split failed or wasn't reasonable

def _search_elements_for_keywords(
    elements: List[Dict[str, Any]],
    form_keywords: List[str],
    schedule_keywords: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Searches a list of elements for Form and Schedule keywords."""
    found_form_title = None
    found_schedule_title = None

    for element in elements:
        if not isinstance(element, dict) or 'text' not in element:
            logging.warning(f"Skipping invalid element during keyword search: {element}")
            continue
        text = element['text']
        text_stripped = text.strip()
        contains_form = False
        contains_schedule = False

        # Check for Form keyword
        for keyword in form_keywords:
            if re.search(rf"\b{keyword}\b", text):
                contains_form = True
                break

        # Check for Schedule keyword
        for keyword in schedule_keywords:
            if re.search(rf"\b{keyword}\b", text):
                contains_schedule = True
                break

        # Assign titles based on findings
        if contains_form and contains_schedule:
            # Both keywords in the same element, attempt split
            split_result = _attempt_split_title(text, schedule_keywords)
            if split_result:
                # Assign only if global titles are not already set
                if found_form_title is None: found_form_title = split_result[0]
                if found_schedule_title is None: found_schedule_title = split_result[1]
            elif found_form_title is None:
                # If split failed, assign the whole text to form_title as priority
                found_form_title = text_stripped
                logging.warning(f"Could not split combined title, assigned full text to form_title: {found_form_title}")

        elif contains_form:
            if found_form_title is None: # Assign only if not already found
                found_form_title = text_stripped

        elif contains_schedule:
            if found_schedule_title is None: # Assign only if not already found
                found_schedule_title = text_stripped

        # Optimization: If both found, stop searching this list
        if found_form_title is not None and found_schedule_title is not None:
            break

    return found_form_title, found_schedule_title

def _apply_header_fallback(
    header_elements: List[Dict[str, Any]],
    schedule_keywords: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Applies fallback logic: uses the topmost header as form title if available."""
    fallback_form_title = None
    fallback_schedule_title = None

    if header_elements:
        first_header = header_elements[0]
        if isinstance(first_header, dict) and 'text' in first_header:
            logging.debug("Applying header fallback (topmost header element).")
            fallback_form_title = first_header['text'].strip()
            # Check if this fallback title *also* contains "Schedule"
            split_result = _attempt_split_title(fallback_form_title, schedule_keywords)
            if split_result:
                fallback_form_title, fallback_schedule_title = split_result
        else:
            logging.warning("Could not apply header fallback title: Invalid first header element.")

    return fallback_form_title, fallback_schedule_title

def _search_body_for_schedule_only(
    body_phrases: List[Dict[str, Any]],
    schedule_keywords: List[str],
    current_form_title: Optional[str]
) -> Optional[str]:
    """Performs a final scan of body phrases ONLY for schedule keywords."""
    logging.debug("Doing final scan of body for separate Schedule title...")
    for phrase in body_phrases:
        if not isinstance(phrase, dict) or 'text' not in phrase: continue
        text = phrase['text']
        text_stripped = text.strip()
        # Skip the phrase if it was already identified as the form title
        if current_form_title and text_stripped == current_form_title: continue

        for keyword in schedule_keywords:
            if re.search(rf"\b{keyword}\b", text):
                logging.info(f"Found separate schedule title in body: '{text_stripped}'")
                return text_stripped # Return the first match found
    return None

def extract_form_schedule_titles(
    header_elements: List[Dict[str, Any]],
    body_phrases: List[Dict[str, Any]]
) -> Dict[str, Optional[str]]:
    """Identifies Form and Schedule titles using headers and fallback to body text.

    Prioritizes headers, then uses body text only if the form title isn't found
    in headers. A missing schedule title is acceptable.

    Args:
        header_elements: List of text elements identified as headers.
        body_phrases: List of text elements identified as body text.

    Returns:
        A dictionary with 'form_title' and 'schedule_title'.
    """
    form_title: Optional[str] = None
    schedule_title: Optional[str] = None
    form_keywords = ["Form", "form"]
    schedule_keywords = ["Schedule", "SCHEDULE"]

    # 1. Search Headers by Keyword
    header_elements.sort(key=lambda x: (x.get('position', [0,0,0,0])[1], -x.get('size', 0)))
    form_title, schedule_title = _search_elements_for_keywords(
        header_elements, form_keywords, schedule_keywords
    )
    found_in_headers = form_title is not None

    # 2. Apply Header Fallback if Form Title still missing
    if form_title is None:
        fb_form, fb_schedule = _apply_header_fallback(header_elements, schedule_keywords)
        if fb_form:
            form_title = fb_form
            # Only use fallback schedule if main search didn't find one
            if schedule_title is None:
                schedule_title = fb_schedule

    # 3. Search Body ONLY if Form Title is STILL missing
    if form_title is None:
        logging.info("Form title not found in headers (incl. fallback), searching body...")
        body_phrases.sort(key=lambda x: (x.get('position', [0,0,0,0])[1], x.get('position', [0,0,0,0])[0]))
        # Search body for *both* keywords, prioritizing form title assignment
        body_form, body_schedule = _search_elements_for_keywords(
            body_phrases, form_keywords, schedule_keywords
        )
        if body_form:
            form_title = body_form
            # Only use body schedule if main/fallback search didn't find one
            if schedule_title is None:
                schedule_title = body_schedule

    # 4. Final Body Scan for Schedule if Still Missing (and Form was found somewhere)
    if form_title is not None and schedule_title is None:
        schedule_title = _search_body_for_schedule_only(
            body_phrases, schedule_keywords, form_title
        )

    # Final Logging
    search_location = "headers"
    if not found_in_headers and form_title is not None:
        search_location = "body (fallback)"
    elif not found_in_headers and form_title is None:
         search_location = "headers and body (not found)"

    logging.info(f"Final Titles: Form='{form_title}', Schedule='{schedule_title}' (Primary source: {search_location})")
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

    text_content = " ".join([p.get('text', '') for p in all_text_phrases]).lower()

    # Check for keywords in the combined text
    if "number of individual returns" in text_content or "number of returns" in text_content or "individuals" in text_content:
        logging.debug("Found 'individuals/returns' keyword in text.")
        return AmountUnit.INDIVIDUALS
    elif "number of forms" in text_content:
         logging.debug("Found 'number of forms' keyword in text.")
         return AmountUnit.FORMS
    # Add more specific dollar checks? "thousands of dollars", "amount" etc.
    elif "dollars" in text_content or "amount" in text_content:
        logging.debug("Found 'dollars/amount' keyword in text.")
        return AmountUnit.DOLLARS
    # Check patterns as fallback - might be more precise if keywords are ambiguous
    else:
        logging.debug("No clear keywords found, checking regex patterns...")
        for element in all_text_phrases:
            text = element.get('text', '') # Use 'text' key
            if not text: continue
            if individuals_pattern.search(text): return AmountUnit.INDIVIDUALS
            if dollars_pattern.search(text): return AmountUnit.DOLLARS
            if forms_pattern.search(text): return AmountUnit.FORMS

    logging.warning("Could not determine amount unit from text phrases.")
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
    based on pattern, font size, and position. Classifies remaining phrases as
    headers or body based on font size threshold.

    Args:
        page: The pdfplumber Page object.
        vertical_tolerance: Max vertical distance between word tops for same phrase.
        horizontal_tolerance: Max horizontal distance between words for same phrase.
        line_item_font_size_threshold: Font size below which a number is considered a line item.
        header_font_size_threshold: Font size >= which a phrase is considered a header.
        page_corner_threshold_pct: Percentage of page dimensions to filter corners.

    Returns:
        A dictionary containing 'line_item_numbers', 'header_phrases', 'body_phrases'.
    """
    # --- Extract words with size and sort ---
    words = page.extract_words(extra_attrs=['fontname', 'size'])
    if not words:
        logging.warning(f"Page {page.page_number}: No words extracted.")
        return {"line_item_numbers": [], "header_phrases": [], "body_phrases": []}

    # --- Filter corner words ---
    page_width = page.width
    page_height = page.height
    corner_width = page_width * page_corner_threshold_pct
    corner_height = page_height * page_corner_threshold_pct

    filtered_words = []
    for word in words:
        x0, top, x1, bottom = word['x0'], word['top'], word['x1'], word['bottom']
        is_in_corner = (
            (x0 < corner_width and top < corner_height) or # Top-left
            (x1 > page_width - corner_width and top < corner_height) or # Top-right
            (x0 < corner_width and bottom > page_height - corner_height) or # Bottom-left
            (x1 > page_width - corner_width and bottom > page_height - corner_height) # Bottom-right
        )
        # Also filter very large text that spans most of the page width?
        # if word['size'] > 30 and (x1 - x0) > page_width * 0.8: continue
        if not is_in_corner:
            filtered_words.append(word)

    if not filtered_words:
         logging.warning(f"Page {page.page_number}: No words left after corner filtering.")
         return {"line_item_numbers": [], "header_phrases": [], "body_phrases": []}

    # --- Group words into phrases (simplified approach) ---
    # Sort primarily by top, then left
    filtered_words.sort(key=lambda w: (w['top'], w['x0']))
    phrases_data = [] # Store {'text': str, 'position': tuple, 'size': float}
    current_phrase_words = []

    for i, word in enumerate(filtered_words):
        if not current_phrase_words:
            current_phrase_words.append(word)
        else:
            last_word = current_phrase_words[-1]
            # Check vertical alignment (allow small tolerance)
            vert_aligned = abs(word['top'] - last_word['top']) < vertical_tolerance
            # Check horizontal proximity (allow slightly negative overlap for tolerance)
            horz_close = (word['x0'] - last_word['x1']) < horizontal_tolerance

            if vert_aligned and horz_close:
                current_phrase_words.append(word)
            else:
                # Finalize the previous phrase
                if current_phrase_words:
                    text = " ".join(w['text'] for w in current_phrase_words)
                    pos = (
                        min(w['x0'] for w in current_phrase_words),
                        min(w['top'] for w in current_phrase_words),
                        max(w['x1'] for w in current_phrase_words),
                        max(w['bottom'] for w in current_phrase_words)
                    )
                    # Use size of the first word as representative size for the phrase
                    size = current_phrase_words[0].get('size')
                    phrases_data.append({"text": text, "position": pos, "size": size})
                # Start new phrase
                current_phrase_words = [word]

    # Add the last phrase
    if current_phrase_words:
         text = " ".join(w['text'] for w in current_phrase_words)
         pos = (
             min(w['x0'] for w in current_phrase_words),
             min(w['top'] for w in current_phrase_words),
             max(w['x1'] for w in current_phrase_words),
             max(w['bottom'] for w in current_phrase_words)
         )
         size = current_phrase_words[0].get('size')
         phrases_data.append({"text": text, "position": pos, "size": size})

    # --- Separate Line Items, Headers, Body ---
    line_items = []
    header_phrases = []
    body_phrases = []
    # Regex for line item numbers (e.g., "1", "1a", "1.", "1a.") at the start of a phrase
    # Requires a word boundary after the number/letter part.
    line_num_pattern = re.compile(r"^\s*(\d+[a-z]?)\.?\b", re.IGNORECASE)
    # Regex to check if a phrase ONLY contains the line item number pattern
    only_line_num_pattern = re.compile(r"^\s*(\d+[a-z]?)\.?\s*$", re.IGNORECASE)

    for phrase in phrases_data:
        text = phrase['text']
        size = phrase.get('size')
        pos = phrase['position']

        match = line_num_pattern.match(text)
        only_match = only_line_num_pattern.match(text)

        # Condition 1: Line Item Number?
        # - Matches pattern at start.
        # - AND (Is the ONLY thing in the phrase OR Font size is below threshold)
        # - AND Position is somewhat to the left.
        is_potential_line_item = False
        if match and pos[0] < page_width * 0.2: # Left position check
             if only_match: # If the phrase is ONLY the number, it's likely a line item
                 is_potential_line_item = True
             elif size is not None and size < line_item_font_size_threshold: # Or if size is small
                 is_potential_line_item = True

        if is_potential_line_item:
             line_num_text = match.group(1)
             # Add more info to line item dict
             line_items.append({
                 'line_item_number': line_num_text,
                 'position': pos,
                 'size': size,
                 'text': text # Keep original text for context
             })
        # Condition 2: Header? (Size >= threshold and not classified as line item)
        elif size is not None and size >= header_font_size_threshold:
             header_phrases.append(phrase)
        # Condition 3: Body phrase (everything else)
        else:
             body_phrases.append(phrase)

    # --- Sort line items ---
    def get_sort_key(item):
        num_str = item['line_item_number']
        match_num = re.match(r"(\d+)([a-z]*)", num_str, re.IGNORECASE)
        if match_num:
            return (int(match_num.group(1)), match_num.group(2).lower())
        else:
            return (float('inf'), num_str) # Put non-standard formats last

    line_items.sort(key=get_sort_key)

    logging.debug(f"Page {page.page_number}: Extracted {len(line_items)} line items, {len(header_phrases)} headers, {len(body_phrases)} body phrases.")
    return {
        "line_item_numbers": line_items,
        "header_phrases": header_phrases,
        "body_phrases": body_phrases
    }

def associate_line_labels(
    line_items: List[Dict[str, Any]],
    text_phrases: List[Dict[str, Any]], # Typically body_phrases
    default_vertical_tolerance: float = 5.0
) -> List[Dict[str, Any]]:
    """Associates descriptive labels (text phrases) to line item numbers.

    Simple approach: Find the closest text phrase to the right of the line number
    on roughly the same vertical level.

    Args:
        line_items: List of identified line item dicts (need 'line_item_number', 'position').
        text_phrases: List of body text phrase dicts (need 'text', 'position').
        default_vertical_tolerance: How much vertical overlap/proximity is allowed.

    Returns:
        A list of dictionaries, extending line_items with 'label', 'label_position',
        and 'combined_position'. Items with no associated label will have 'label' set to None.
    """
    labeled_lines = []
    used_phrase_indices = set()

    for item in line_items:
        item_num = item['line_item_number']
        item_pos = item['position'] # [x0, top, x1, bottom]
        item_y_center = (item_pos[1] + item_pos[3]) / 2
        item_height = item_pos[3] - item_pos[1]
        v_tolerance = max(item_height / 2, default_vertical_tolerance) # Use at least default

        best_match_phrase = None
        min_distance = float('inf')
        best_match_idx = -1

        for i, phrase in enumerate(text_phrases):
            if i in used_phrase_indices:
                continue

            phrase_pos = phrase['position']
            phrase_y_center = (phrase_pos[1] + phrase_pos[3]) / 2

            # 1. Vertical Alignment Check
            if abs(item_y_center - phrase_y_center) < v_tolerance:
                # 2. Horizontal Position Check (Phrase must be to the right of the item number)
                # Check start of phrase vs end of item number
                if phrase_pos[0] >= item_pos[2]:
                    distance = phrase_pos[0] - item_pos[2] # Horizontal gap
                    if distance < min_distance:
                        # Basic overlap check (optional, maybe not needed if using center align)
                        # vert_overlap = max(0, min(item_pos[3], phrase_pos[3]) - max(item_pos[1], phrase_pos[1]))
                        # if vert_overlap > 0: # Ensure some vertical overlap if using strict top/bottom
                        min_distance = distance
                        best_match_phrase = phrase
                        best_match_idx = i
                # Consider phrases that *start* slightly left but overlap significantly? More complex.

        label_text = None
        label_pos = None
        combined_pos = list(item_pos) # Default to item pos if no label found

        if best_match_phrase:
            label_text = best_match_phrase['text']
            label_pos = best_match_phrase['position']
            used_phrase_indices.add(best_match_idx)
            # Combine bounding box
            combined_pos = (
                min(item_pos[0], label_pos[0]),
                min(item_pos[1], label_pos[1]),
                max(item_pos[2], label_pos[2]),
                max(item_pos[3], label_pos[3])
            )

        labeled_lines.append({
            **item, # Include original line item info (number, pos, size, text)
            "label": label_text,
            "label_position": label_pos,
            "combined_position": combined_pos
        })

    found_labels_count = sum(1 for line in labeled_lines if line['label'] is not None)
    logging.debug(f"Associated labels for {found_labels_count}/{len(line_items)} line items.")
    return labeled_lines

def associate_amounts_multimodal(
    page: pdfplumber.page.Page,
    labeled_lines: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    model_name: str = "gemini-1.5-flash-latest"
) -> Dict[str, Optional[str]]:
    """Associates amounts with line items using a multimodal LLM.

    Args:
        page: The pdfplumber Page object.
        labeled_lines: List of dicts with 'line_item_number', 'label', 'combined_position'.
        amounts: List of dicts with 'amount' and 'position'.
        model_name: The name of the multimodal model to use.

    Returns:
        A dictionary mapping line_item_number (str) to amount_text (str or None).
    """
    # Dynamically import necessary libraries only when this function is called
    try:
        # from ai_tax_agent.settings import settings # Already imported globally
        # from langchain_google_genai import ChatGoogleGenerativeAI # Already imported globally
        # from langchain_core.messages import HumanMessage, SystemMessage # Already imported globally
        from PIL import Image # Pillow for image handling
    except ImportError as e:
        logging.error(f"Missing required libraries for multimodal processing: {e}. Please install Pillow.")
        return {line['line_item_number']: None for line in labeled_lines if line.get("label") is not None}


    amount_map = {}
    for line_data in labeled_lines:
        if line_data.get("label") is not None:
            amount_map[line_data['line_item_number']] = None

    if not amounts:
        logging.info(f"Page {page.page_number}: No amounts found, skipping multimodal association.")
        return amount_map
    if not amount_map:
        logging.info(f"Page {page.page_number}: No labeled line items found, skipping multimodal association.")
        return {}


    try:
        # 1. Convert page to image
        img = page.to_image(resolution=150)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        # 2. Prepare input data for prompt
        prompt_lines = []
        for line in labeled_lines:
             if line.get("label") is not None:
                pos = line.get('combined_position') or line.get('line_item_position')
                prompt_lines.append({
                    "ln": line['line_item_number'],
                    "lbl": line['label'],
                    "pos": [round(p, 1) for p in pos] if pos else None
                })

        prompt_amounts = []
        for amount in amounts:
             pos = amount.get('position')
             prompt_amounts.append({
                 "amt": amount['amount'],
                 "pos": [round(p, 1) for p in pos] if pos else None
             })

        # 3. Construct Prompt
        system_prompt = "You are an expert assistant analyzing PDF form structures. Your task is to associate numeric amounts with the correct line item based on their positions in the provided image and data."
        prompt_lines_json = json.dumps(prompt_lines, indent=2)
        prompt_amounts_json = json.dumps(prompt_amounts, indent=2)

        human_prompt_text = f"""
Analyze the provided image of a form page.
Here is a list of identified line items with their labels and approximate bounding boxes [x0, top, x1, bottom]:
```json
{prompt_lines_json}
```

Here is a list of identified numeric amounts and their bounding boxes:
```json
{prompt_amounts_json}
```
Note: These numeric amounts often appear in a distinct blue color in the image.

Based on the visual layout in the image, associate each amount with the single most likely line item number it corresponds to. Consider typical form layouts where amounts appear in columns to the right of or below labels.

Output ONLY a JSON object mapping the line_item_number (string) to the corresponding amount (string).
If a line item number from the input list does not have an associated amount in the image, its value should be null in the output JSON.
If an amount cannot be confidently associated with any line item, omit it from the output map.
Example output format: {{"1": "123,456", "2a": "789", "3": null}}
"""

        # 4. Initialize and Call LLM
        if not settings.gemini_api_key:
             logging.error("GEMINI_API_KEY not found in settings.")
             return amount_map

        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=settings.gemini_api_key, temperature=0.1)
        logging.info(f"Page {page.page_number}: Calling multimodal LLM ({model_name}) for amount association...")
        message = HumanMessage(
            content=[
                {"type": "text", "text": human_prompt_text},
                {"type": "image_url", "image_url": {"url": img_data_uri}},
            ]
        )
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        response_content = response.content
        logging.debug(f"LLM Response received (page {page.page_number}): {response_content[:500]}...")

        # 5. Parse Response
        try:
            logging.debug(f"Attempting to find JSON in LLM response:\n{response_content}")
            json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response_content, re.DOTALL | re.IGNORECASE)
            parsed_map = None

            if json_match:
                json_string = json_match.group(1).strip()
                logging.debug(f"Found JSON block, attempting to parse: {json_string[:100]}...")
                if json_string:
                    try: parsed_map = json.loads(json_string)
                    except json.JSONDecodeError as e: logging.error(f"Failed to parse extracted JSON string: {e}\nString: {json_string}")
                else: logging.warning("Regex matched JSON block, but captured string was empty.")
            else:
                logging.warning("Could not find JSON block ```json...``` in response. Attempting to parse entire response.")
                try:
                    cleaned_response = response_content.strip()
                    if cleaned_response: parsed_map = json.loads(cleaned_response)
                    else: logging.warning("LLM response content was empty after stripping whitespace.")
                except json.JSONDecodeError as e: logging.error(f"Failed to parse entire LLM response as JSON: {e}\nResponse: {response_content}")

            # Validate and update
            if parsed_map and isinstance(parsed_map, dict):
                 update_count = 0
                 for ln, amt in parsed_map.items():
                     if ln in amount_map:
                          amount_map[ln] = str(amt) if amt is not None else None
                          update_count += 1
                     else: logging.warning(f"LLM returned amount for unexpected line number: {ln}")
                 logging.info(f"Page {page.page_number}: Successfully parsed and applied LLM amount map for {update_count} lines.")
            else:
                if parsed_map is not None: logging.error(f"Parsed LLM response was not dict: type={type(parsed_map)}, content={str(parsed_map)[:200]}...")
        except Exception as e:
            logging.error(f"Error processing LLM response content: {e}\nResponse: {response_content}", exc_info=True)

    except ImportError:
         logging.error("Pillow library not found. Please install it (`pip install Pillow`).")
    except Exception as e:
        logging.error(f"Page {page.page_number}: Error during multimodal amount association: {e}", exc_info=True)

    return amount_map

# --- Function to extract structure using LLM --- 
def extract_structure_multimodal(
    page: pdfplumber.page.Page,
    model_name: str = "gemini-1.5-flash-latest"
) -> Dict[str, Any]:
    """Extracts form/schedule titles and line items using a multimodal LLM.

    Args:
        page: The pdfplumber Page object.
        model_name: The name of the multimodal model to use.

    Returns:
        A dictionary containing:
            'form_title': Extracted form title (str or None).
            'schedule_title': Extracted schedule title (str or None).
            'line_items': List of dicts, each with 'line_item_number' and 'label'.
                          Returns empty list if parsing fails or none found.
    """
    # Default structure in case of errors
    result = {"form_title": None, "schedule_title": None, "line_items": []}

    try:
        from PIL import Image # Ensure Pillow is available
    except ImportError as e:
        logging.error(f"Missing required library Pillow for multimodal structure extraction: {e}. Please install it.")
        return result

    try:
        # 1. Convert page to image
        img = page.to_image(resolution=150)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        # 2. Construct Prompt
        system_prompt = (
            "You are an expert assistant specialized in analyzing PDF tax forms. "
            "Your task is to identify the main form title, any associated schedule title, "
            "and extract all line items (number and label pairs) based on the visual layout in the provided image."
        )

        few_shot_titles = "Form 1099-K", "Form 3921", "Form 3922", "Form 1065, Schedule K"
        line_item_example = '{"line_item_number": "4b", "label": "Guaranteed payments for capital"}'

        human_prompt_text = f"""
Analyze the provided image of a form page.

1.  **Identify Titles:** Determine the main Form Title and any Schedule Title. These are usually prominent text near the top. 
    Examples of expected titles: {json.dumps(few_shot_titles)}.

2.  **Extract Line Items:** Find all line items on the page. Line items typically consist of:
    *   A line number (e.g., '1', '2a', '15c') often in a smaller font, possibly within a box or column on the left.
    *   A corresponding descriptive label text, usually located to the right of or below the line number.
    Extract pairs of line item numbers and their full label text.
    Example line item structure: {line_item_example}

**Output Format:**
Output ONLY a single JSON object containing the following keys:
*   `form_title`: The main form title identified (string or null if none found).
*   `schedule_title`: The schedule title, if present (string or null).
*   `line_items`: A JSON list of objects, where each object has:
    *   `line_item_number`: The extracted line number (string).
    *   `label`: The full label text associated with the line number (string).

Example JSON Output:
```json
{{
  "form_title": "Form 1065",
  "schedule_title": "Schedule K-1",
  "line_items": [
    {{"line_item_number": "1", "label": "Ordinary business income (loss)"}},
    {{"line_item_number": "4a", "label": "Guaranteed payments for services"}},
    {line_item_example}
  ]
}}
```
"""

        # 3. Initialize and Call LLM
        if not settings.gemini_api_key:
             logging.error("GEMINI_API_KEY not found in settings for structure extraction.")
             return result

        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=settings.gemini_api_key, temperature=0.1)
        logging.info(f"Page {page.page_number + 1}: Calling multimodal LLM ({model_name}) for structure extraction (titles, lines)...")

        message = HumanMessage(
            content=[
                {"type": "text", "text": human_prompt_text},
                {"type": "image_url", "image_url": {"url": img_data_uri}},
            ]
        )
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        response_content = response.content
        logging.debug(f"LLM Structure Response received (page {page.page_number + 1}): {response_content[:500]}...")

        # 4. Parse Response (using robust method)
        parsed_data = None
        try:
            json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response_content, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_string = json_match.group(1).strip()
                if json_string:
                    try: parsed_data = json.loads(json_string)
                    except json.JSONDecodeError as e: logging.error(f"Page {page.page_number + 1}: Failed to parse extracted JSON structure string: {e}\nString: {json_string}")
                else: logging.warning(f"Page {page.page_number + 1}: Regex matched JSON block for structure, but captured string was empty.")
            else:
                logging.warning(f"Page {page.page_number + 1}: Could not find JSON block ```json...``` in structure response. Attempting to parse entire response.")
                cleaned_response = response_content.strip()
                if cleaned_response:
                    try: parsed_data = json.loads(cleaned_response)
                    except json.JSONDecodeError as e: logging.error(f"Page {page.page_number + 1}: Failed to parse entire LLM structure response as JSON: {e}\nResponse: {response_content}")
                else: logging.warning(f"Page {page.page_number + 1}: LLM structure response was empty after stripping whitespace.")

            # Basic validation of parsed structure
            if isinstance(parsed_data, dict) and isinstance(parsed_data.get('line_items'), list):
                result["form_title"] = parsed_data.get("form_title")
                result["schedule_title"] = parsed_data.get("schedule_title")
                # Further validate line items list if needed
                validated_lines = []
                for item in parsed_data['line_items']:
                    if isinstance(item, dict) and 'line_item_number' in item and 'label' in item:
                        validated_lines.append({
                            "line_item_number": str(item['line_item_number']), # Ensure string
                            "label": str(item['label']) # Ensure string
                        })
                    else:
                        logging.warning(f"Page {page.page_number + 1}: Skipping invalid line item structure: {item}")
                result["line_items"] = validated_lines
                logging.info(f"Page {page.page_number + 1}: Successfully parsed structure from LLM: Title='{result['form_title']}', Schedule='{result['schedule_title']}', Found {len(result['line_items'])} line items.")
            else:
                logging.error(f"Page {page.page_number + 1}: Parsed LLM structure response was not a valid dictionary or missing 'line_items' list. Parsed: {str(parsed_data)[:200]}...")

        except Exception as e:
            logging.error(f"Page {page.page_number + 1}: Error processing LLM structure response content: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Page {page.page_number + 1}: Error during multimodal structure extraction: {e}", exc_info=True)

    return result

# --- Main Orchestration Function --- 
def parse_pdf_page_structure(
    pdf_path: str,
    page_num_one_indexed: int,
    amount_color_tuple: Tuple[float, float, float] = (0, 0, 0.5), # Example: Dark Blue
    # Heuristic parameters are no longer used directly here but kept for reference
    vertical_tolerance: float = 2,
    horizontal_tolerance: float = 5,
    line_item_font_size_threshold: float = 7.0,
    header_font_size_threshold: float = 7.0,
    page_corner_threshold_pct: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Parses a single PDF page using multimodal LLMs for structure and amounts.

    Orchestrates calls to extract amounts by color, determine amount unit, and then
    uses multimodal LLMs to extract titles, line items, and associate amounts.

    Args:
        (See above functions for parameter details)

    Returns:
        A dictionary containing the parsed structure or None if processing fails.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return None

    page_index = page_num_one_indexed - 1

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index < 0 or page_index >= len(pdf.pages):
                logging.error(f"Page number {page_num_one_indexed} out of range (1-{len(pdf.pages)}).")
                return None

            page = pdf.pages[page_index]
            logging.info(f"--- Processing Page {page.page_number + 1} ---")

            # --- Extract elements NOT replaced by LLM --- 
            # (Amounts by color, potentially Amount Unit determination heuristically)
            logging.debug("Extracting amounts by color...")
            amounts = extract_amounts_by_color(page, amount_color_tuple=amount_color_tuple)
            
            # NOTE: Amount Unit determination MIGHT also be included in the structure LLM call
            #       or kept separate. Keeping separate for now.
            #       Requires phrases, so we need to extract them OR pass all text to LLM.
            #       For simplicity, let's extract phrases just for unit detection for now.
            logging.debug("Extracting phrases for unit detection...")
            # Using a basic word extraction for unit detection context
            all_words = page.extract_words()
            all_text_for_unit = " ".join([w.get('text','') for w in all_words])
            logging.debug("Determining amount unit...")
            amount_unit = determine_amount_unit([{'text': all_text_for_unit}]) # Pass as a single phrase

            # --- Use LLM for Titles and Line Items --- 
            logging.debug("Extracting structure (titles, line items) via multimodal LLM...")
            structure_data = extract_structure_multimodal(page)
            form_title = structure_data.get("form_title")
            schedule_title = structure_data.get("schedule_title")
            # This list now contains dicts with 'line_item_number' and 'label'
            llm_extracted_line_items = structure_data.get("line_items", [])

            # --- Associate Amounts using LLM --- 
            logging.debug("Associating amounts via multimodal LLM...")
            # We need a list compatible with associate_amounts_multimodal's input.
            # It expects 'labeled_lines' with 'line_item_number' and 'label'.
            # The structure_data['line_items'] already has this format.
            # However, the amount associator might benefit from position info if available.
            # For now, just pass the essential info extracted by the structure LLM.
            amount_map = associate_amounts_multimodal(page, llm_extracted_line_items, amounts)

            # --- Build final output --- 
            # Use the line items extracted by the structure LLM
            output_line_items = []
            if isinstance(llm_extracted_line_items, list):
                for line_data in llm_extracted_line_items:
                    # Basic check if item seems valid (already validated in structure extractor)
                    if isinstance(line_data, dict) and 'line_item_number' in line_data and 'label' in line_data:
                        line_num = line_data['line_item_number']
                        output_line_items.append({
                            "line_item_number": line_num,
                            "label": line_data['label'],
                            "amount": amount_map.get(line_num), # Get amount from the second LLM call
                        })
                    else:
                         logging.warning(f"Page {page.page_number + 1}: Found invalid item in list from structure LLM: {line_data}")
            else:
                 logging.error(f"Page {page.page_number + 1}: line_items returned by structure LLM was not a list: {llm_extracted_line_items}")


            final_output = {
                "form_title": form_title, # From structure LLM
                "schedule_title": schedule_title, # From structure LLM
                "amount_unit": amount_unit.value, # From heuristic for now
                "line_items": output_line_items # From structure LLM + amount LLM
            }
            logging.info(f"Successfully parsed structure for page {page.page_number + 1}.")
            return final_output

    except Exception as e:
        logging.error(f"An error occurred during PDF processing for page {page_num_one_indexed}: {e}", exc_info=True)
        return None

def parse_full_pdf_structure(
    pdf_path: str,
    start_page: int = 1
) -> List[Dict[str, Any]]:
    """Parses a PDF document page by page starting from a given page number.

    Iterates through pages, calling parse_pdf_page_structure for each, and aggregates
    the results.

    Args:
        pdf_path: Path to the PDF file.
        start_page: The 1-based page number to start parsing from.

    Returns:
        A list of dictionaries, where each dictionary contains the parsed
        structure for a single page ('page_number', 'form_title', 'schedule_title',
        'amount_unit', 'line_items').
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return []

    all_pages_data = []
    total_pages = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Starting PDF parsing for '{os.path.basename(pdf_path)}' from page {start_page} up to {total_pages}.")

            if start_page < 1 or start_page > total_pages:
                logging.error(f"Start page {start_page} is out of range (1-{total_pages}).")
                return []

            # Iterate from start_page (1-indexed) up to total_pages
            for page_num in range(start_page, total_pages + 1):
                # Logging moved inside parse_pdf_page_structure
                page_data = parse_pdf_page_structure(
                    pdf_path=pdf_path,
                    page_num_one_indexed=page_num
                )

                if page_data:
                    page_data['page_number'] = page_num # Add page number to the result
                    all_pages_data.append(page_data)
                else:
                    logging.warning(f"Skipping page {page_num} due to parsing errors or no structure found.")

    except pdfplumber.exceptions.PDFSyntaxError as pdf_err:
         logging.error(f"Failed to open or parse PDF '{pdf_path}': {pdf_err}", exc_info=True)
         return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while opening or iterating through '{pdf_path}': {e}", exc_info=True)
        return []

    processed_count = len(all_pages_data)
    attempted_count = total_pages - start_page + 1 if total_pages >= start_page else 0
    logging.info(f"Finished parsing '{os.path.basename(pdf_path)}'. Successfully processed {processed_count}/{attempted_count} pages (starting from page {start_page}).")
    return all_pages_data