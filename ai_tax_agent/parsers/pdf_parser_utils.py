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

def extract_form_schedule_titles(
    header_elements: List[Dict[str, Any]],
    body_phrases: List[Dict[str, Any]] # Add body phrases as input
) -> Dict[str, Optional[str]]:
    """Identifies Form and Schedule titles from header and optionally body elements.

    Searches headers first, then body text if titles are not found in headers.
    Assumes titles are usually larger font text near the top in headers.

    Args:
        header_elements: List of text elements identified as headers.
        body_phrases: List of text elements identified as body text.

    Returns:
        A dictionary with 'form_title' and 'schedule_title'.
    """
    form_title = None
    schedule_title = None

    # --- Search Headers First ---
    # Sort by vertical position (top first), then maybe font size (largest first)
    header_elements.sort(key=lambda x: (x.get('position', [0,0,0,0])[1], -x.get('size', 0)))

    form_keywords = ["Form", "form"]
    schedule_keywords = ["Schedule", "SCHEDULE"] # Keep case variations

    found_form_in_header = False
    found_schedule_in_header = False

    for element in header_elements:
        # Ensure element is a dict and has 'text' key
        if not isinstance(element, dict) or 'text' not in element:
            logging.warning(f"Skipping invalid header element: {element}")
            continue
        text = element['text']

        # Prioritize finding Form first
        if not found_form_in_header:
            for keyword in form_keywords:
                # Use regex for word boundary to avoid matching substrings like "Information"
                if re.search(rf"\\b{keyword}\\b", text):
                    form_title = text.strip()
                    found_form_in_header = True
                    break # Found form keyword in this element

        # Look for Schedule in the same element or others
        # Don't reset found_schedule_in_header if already found
        if not found_schedule_in_header:
             for keyword in schedule_keywords:
                 if re.search(rf"\\b{keyword}\\b", text):
                     # Avoid assigning the same text to both if keywords overlap/present in same string initially
                     if text.strip() != form_title:
                         schedule_title = text.strip()
                         found_schedule_in_header = True
                         break # Found schedule keyword in this element
                     # Handle case where both keywords are in the same header element (e.g. "Form 1040 Schedule A")
                     elif found_form_in_header and form_title == text.strip():
                          # Attempt to split
                          parts = re.split(rf"(\\b{keyword}\\b.*)", text, 1, re.IGNORECASE)
                          if len(parts) >= 3: # Should get ["Form xxx ", "Schedule yyy", ""] potentially
                              form_title = parts[0].strip()
                              # Reconstruct schedule part carefully
                              schedule_title = "".join(parts[1:]).strip()
                              found_schedule_in_header = True
                              break # Found and split schedule

        # If we found distinct titles for both, we can stop searching headers
        if found_form_in_header and found_schedule_in_header and form_title != schedule_title:
            break

    # Fallback: if no keywords found in headers, assign the topmost largest element as form title
    if not found_form_in_header and header_elements:
        # Ensure the first element is valid before accessing 'text'
        first_header = header_elements[0]
        if isinstance(first_header, dict) and 'text' in first_header:
             form_title = first_header['text'].strip()
             # Check if this fallback assignment contains "Schedule"
             if not found_schedule_in_header:
                  for keyword in schedule_keywords:
                      if re.search(rf"\\b{keyword}\\b", form_title):
                           # Attempt split again if fallback contained schedule
                           parts = re.split(rf"(\\b{keyword}\\b.*)", form_title, 1, re.IGNORECASE)
                           if len(parts) >= 3:
                               form_title = parts[0].strip()
                               schedule_title = "".join(parts[1:]).strip()
                               found_schedule_in_header = True
                               break
        else:
            logging.warning("Could not apply header fallback title: Invalid first header element.")


    # --- Search Body if Titles Still Missing ---
    # Search body only if we didn't find a specific title using keywords in headers
    # or the fallback didn't provide one.
    search_body_for_form = not form_title
    search_body_for_schedule = not schedule_title

    if search_body_for_form or search_body_for_schedule:
        logging.debug("Searching body phrases for missing titles...")
        # Sort body phrases? Maybe just scan is fine. Sort by position might help.
        body_phrases.sort(key=lambda x: (x.get('position', [0,0,0,0])[1], x.get('position', [0,0,0,0])[0]))
        for phrase in body_phrases:
             # Ensure phrase is a dict and has 'text' key
             if not isinstance(phrase, dict) or 'text' not in phrase:
                 logging.warning(f"Skipping invalid body phrase: {phrase}")
                 continue
             text = phrase['text']

             if search_body_for_form:
                 for keyword in form_keywords:
                      if re.search(rf"\\b{keyword}\\b", text):
                           # Simple assignment from body - might be less reliable
                           form_title = text.strip()
                           search_body_for_form = False # Found it
                           logging.debug(f"Found potential form title in body: {form_title}")
                           break

             if search_body_for_schedule:
                  for keyword in schedule_keywords:
                      if re.search(rf"\\b{keyword}\\b", text):
                           # Check if it's the same as the potential form title found in body
                           if text.strip() != form_title:
                               schedule_title = text.strip()
                               search_body_for_schedule = False # Found it
                               logging.debug(f"Found potential schedule title in body: {schedule_title}")
                               break
                           # Handle case where both keywords in same body phrase
                           elif not search_body_for_form and form_title == text.strip():
                                parts = re.split(rf"(\\b{keyword}\\b.*)", text, 1, re.IGNORECASE)
                                if len(parts) >= 3:
                                     form_title = parts[0].strip()
                                     schedule_title = "".join(parts[1:]).strip()
                                     search_body_for_schedule = False
                                     logging.debug(f"Split form/schedule title found in body: {form_title} / {schedule_title}")
                                     break


             # Stop searching body if both found
             if not search_body_for_form and not search_body_for_schedule:
                  break


    logging.info(f"Identified Form Title: '{form_title}', Schedule Title: '{schedule_title}' (Searched headers {'and body' if not (found_form_in_header and found_schedule_in_header) else ''})")
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
                "text": phrase_text,
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

        label_text = best_match_phrase_data['text'] if best_match_phrase_data else None
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

def associate_amounts_multimodal(
    page: pdfplumber.page.Page,
    labeled_lines: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    # Add model name parameter if needed
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
    amount_map = {} 
    # Initialize all line items with None amount
    for line_data in labeled_lines:
        if line_data.get("label") is not None: # Only consider lines with labels
            amount_map[line_data['line_item_number']] = None 

    if not amounts: # No amounts found on page
        logging.info("No amounts found on page, skipping multimodal association.")
        return amount_map

    try:
        # 1. Convert page to image
        img = page.to_image(resolution=150) # Use adequate resolution
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        # 2. Prepare input data for prompt (only essential info)
        prompt_lines = []
        for line in labeled_lines:
             if line.get("label") is not None: # Ensure label exists
                prompt_lines.append({
                    "ln": line['line_item_number'],
                    "lbl": line['label'],
                    # Simplify position for prompt clarity? Use combined or line item?
                    "pos": line.get('combined_position') or line.get('line_item_position')
                })

        prompt_amounts = []
        for amount in amounts:
             prompt_amounts.append({
                 "amt": amount['amount'],
                 "pos": amount['position']
             })

        # 3. Construct Prompt
        system_prompt = "You are an expert assistant analyzing PDF form structures. Your task is to associate numeric amounts with the correct line item based on their positions in the provided image and data."
        human_prompt_text = f"""
Analyze the provided image of a form page.
Here is a list of identified line items with their labels and approximate bounding boxes [x0, y0, x1, y1]:
```json
{json.dumps(prompt_lines, indent=2)}
```

Here is a list of identified numeric amounts and their bounding boxes:\n```json\n{json.dumps(prompt_amounts, indent=2)}\n```\n\n+Note: These numeric amounts often appear in a distinct blue color in the image.\n\nBased on the visual layout in the image, associate each amount with the single most likely line item number it corresponds to. Consider typical form layouts where amounts appear in columns to the right of or below labels.\n\nOutput ONLY a JSON object mapping the line_item_number (string) to the corresponding amount (string). \nIf a line item number from the input list does not have an associated amount in the image, its value should be null in the output JSON. \nIf an amount cannot be confidently associated with any line item, omit it from the output map.\n-Example output format: {{"1\": \"123,456\", \"2a\": \"789\", \"3\": null, ...}}\n+Example output format: {{"1\": \"123,456\", \"2a\": \"789\", \"3\": null, ...}} <-- Ensure double braces here
"""

        # Create the message for the LLM
        message = HumanMessage(
            content=[
                {"type": "text", "text": human_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_uri},
                },
            ]
        )

        # 4. Initialize and Call LLM
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=settings.gemini_api_key, temperature=0.1)
        logging.info(f"Calling multimodal LLM ({model_name}) for amount association...")
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        response_content = response.content
        logging.info(f"LLM Response received: {response_content[:200]}...") # Log snippet

        # 5. Parse Response
        # Expecting a JSON string within the response content
        try:
            # Attempt to find JSON block within potential markdown
            logging.debug(f"Attempting to find JSON in LLM response:\n{response_content}")
            json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response_content, re.DOTALL | re.IGNORECASE)
            parsed_map = None

            if json_match:
                json_string = json_match.group(1).strip()
                logging.debug(f"Found JSON block, attempting to parse: {json_string[:100]}...")
                if json_string:
                    try:
                        parsed_map = json.loads(json_string)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse extracted JSON string: {e}\nString: {json_string}")
                else:
                    logging.warning("Regex matched JSON block, but captured string was empty.")
            else:
                logging.warning("Could not find JSON block ```json...``` in response. Attempting to parse entire response.")
                # Try parsing directly if no markdown found
                try:
                    parsed_map = json.loads(response_content.strip())
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse entire LLM response as JSON: {e}\nResponse: {response_content}")
            
            # Validate and update the initial amount_map
            if parsed_map and isinstance(parsed_map, dict):
                 for ln, amt in parsed_map.items():
                     if ln in amount_map: # Check if it's a valid line number from input
                          amount_map[ln] = amt # Update with LLM result (could be str or None)
                 logging.info("Successfully parsed amount map from LLM response.")
            else:
                if parsed_map is not None:
                     logging.error(f"Parsed LLM response was not a valid dictionary: type={type(parsed_map)}, content={str(parsed_map)[:200]}...")
                # Error already logged if parsing failed

        except Exception as e:
            logging.error(f"Error processing LLM response: {e}\nResponse: {response_content}")

    except ImportError:
         logging.error("Pillow library not found. Please install it (`pip install Pillow`) for image processing.")
    except Exception as e:
        logging.error(f"An error occurred during multimodal amount association: {e}", exc_info=True)

    # Return the map, potentially partially filled or empty if errors occurred
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
            title_info = extract_form_schedule_titles(header_phrases, body_phrases)

            # --- Associate labels to line items --- 
            labeled_lines = associate_line_labels(line_items, body_phrases)

            # --- Associate amounts to labeled lines --- 
            amount_map = associate_amounts_multimodal(page, labeled_lines, amounts)

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
        logging.getLogger("pdfminer").setLevel(logging.ERROR) # Keep pdfminer quiet
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Starting PDF parsing for '{os.path.basename(pdf_path)}' from page {start_page} up to {total_pages}.")

            if start_page < 1 or start_page > total_pages:
                logging.error(f"Start page {start_page} is out of range (1-{total_pages}).")
                return []

            # Iterate from start_page (1-indexed) up to total_pages
            for page_num in range(start_page, total_pages + 1):
                logging.info(f"Processing page {page_num}/{total_pages}...")
                page_data = parse_pdf_page_structure(pdf_path, page_num)

                if page_data:
                    page_data['page_number'] = page_num # Add page number to the result
                    all_pages_data.append(page_data)
                else:
                    logging.warning(f"Skipping page {page_num} due to parsing errors or no structure found.")

    except Exception as e:
        logging.error(f"An error occurred while opening or iterating through '{pdf_path}': {e}", exc_info=True)
        # Optionally return partially collected data or empty list
        # return all_pages_data
        return []

    logging.info(f"Finished parsing. Successfully processed {len(all_pages_data)} pages out of {total_pages - start_page + 1} attempted.")
    return all_pages_data