import argparse
import logging
import os
import sys
import pdfplumber
from typing import Any, Dict, List, Optional
import json

# Add project root to Python path if needed (adjust based on your project structure)
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse a PDF file, analyzing layout elements like text format, position, and color.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    # Add argument for specific page (1-indexed)
    parser.add_argument("--page", type=int, help="Specific 1-indexed page number to process.")
    # Add more arguments as needed, e.g., output path
    # parser.add_argument("-o", "--output", help="Path to save structured output (e.g., JSON).")
    return parser.parse_args()

def is_word_inside_rect(word: Dict[str, Any], rect: Dict[str, Any], tolerance: float = 1.0) -> bool:
    """Checks if a word's bounding box is (mostly) inside a rectangle's bounding box.
    
    Args:
        word: A dictionary representing a word object from pdfplumber.
        rect: A dictionary representing a rect object from pdfplumber.
        tolerance: Small tolerance to handle minor overlaps or rounding errors.
        
    Returns:
        True if the word is considered inside the rectangle, False otherwise.
    """
    # pdfplumber coordinates: top is distance from top edge, bottom is distance from top edge
    # So top < bottom numerically.
    word_x0 = word.get("x0", 0)
    word_top = word.get("top", 0)
    word_x1 = word.get("x1", 0)
    word_bottom = word.get("bottom", 0)
    
    rect_x0 = rect.get("x0", 0)
    rect_top = rect.get("top", 0)
    rect_x1 = rect.get("x1", 0)
    rect_bottom = rect.get("bottom", 0)
    
    # Check if the word's horizontal span is within the rectangle's
    horizontally_inside = (rect_x0 - tolerance <= word_x0) and (word_x1 <= rect_x1 + tolerance)
    # Check if the word's vertical span is within the rectangle's
    vertically_inside = (rect_top - tolerance <= word_top) and (word_bottom <= rect_bottom + tolerance)
    
    return horizontally_inside and vertically_inside

def group_words_into_phrases(
    words: List[Dict[str, Any]], 
    x_gap_tolerance: float = 3.0, # Max horizontal distance between words in a phrase
    y_deviation_tolerance: float = 1.0 # Max vertical distance between word baselines
) -> List[Dict[str, Any]]:
    """Groups individual words into phrases based on proximity and formatting."""
    if not words:
        return []

    # Sort words by vertical position (top), then horizontal (x0)
    words.sort(key=lambda w: (w.get('top', 0), w.get('x0', 0)))

    phrases = []
    current_phrase = None

    for i, word in enumerate(words):
        word_text = word.get('text', '')
        if not word_text: # Skip blank words if any slipped through
             continue
             
        can_append = False
        if current_phrase:
            last_word_in_phrase = current_phrase["_internal_last_word"]
            
            # Check vertical alignment (using 'top' baseline)
            y_aligned = abs(word.get('top', 0) - last_word_in_phrase.get('top', 0)) <= y_deviation_tolerance
            
            # Check horizontal proximity
            h_gap = word.get('x0', 0) - last_word_in_phrase.get('x1', 0)
            h_adjacent = 0 <= h_gap <= x_gap_tolerance
            
            # Check formatting match (font name, size, colors)
            # Note: Color comparison might need adjustment if None/tuple variations exist
            format_match = (
                word.get('fontname') == current_phrase.get('fontname') and
                abs(word.get('size', 0) - current_phrase.get('size', 0)) < 0.1 and # Allow minor size difference
                word.get('stroking_color') == current_phrase.get('stroking_color') and
                word.get('non_stroking_color') == current_phrase.get('non_stroking_color')
            )
            
            can_append = y_aligned and h_adjacent and format_match

        if can_append and current_phrase is not None:
            # Append word to the current phrase
            current_phrase["text"] += " " + word_text
            current_phrase["x1"] = word.get('x1') # Update right boundary
            # Keep top/bottom of the first word for simplicity, or adjust if needed:
            # current_phrase["top"] = min(current_phrase["top"], word.get('top'))
            # current_phrase["bottom"] = max(current_phrase["bottom"], word.get('bottom'))
            current_phrase["_internal_last_word"] = word # Update last word reference
        else:
            # Finalize the previous phrase (if one exists)
            if current_phrase:
                del current_phrase["_internal_last_word"] # Remove internal helper key
                phrases.append(current_phrase)
            
            # Start a new phrase
            current_phrase = {
                "text": word_text,
                "x0": word.get('x0'),
                "top": word.get('top'),
                "x1": word.get('x1'),
                "bottom": word.get('bottom'),
                "fontname": word.get('fontname'),
                "size": word.get('size'),
                "stroking_color": word.get('stroking_color'),
                "non_stroking_color": word.get('non_stroking_color'),
                "_internal_last_word": word # Helper to track last word added
            }

    # Add the last phrase after the loop finishes
    if current_phrase:
        del current_phrase["_internal_last_word"]
        phrases.append(current_phrase)
        
    logger.info(f"Grouped {len(words)} words into {len(phrases)} phrases.")
    return phrases

def extract_numbered_labels(
    phrases: List[Dict[str, Any]],
    target_fontname: str,
    target_size_min: float,
    target_size_max: float,
    max_h_gap: float = 5.0, # Max horizontal distance between number and label
    max_v_align_diff: float = 1.0 # Max vertical difference in 'top' baseline
) -> List[Dict[str, str]]:
    """Finds phrases matching target format that have a numeric phrase immediately to their left."""
    
    numbered_labels = []
    potential_labels = []
    potential_numbers = []

    # Separate potential labels and numbers
    for phrase in phrases:
        # Check if phrase matches target label format
        p_font = phrase.get('fontname')
        p_size = phrase.get('size', 0)
        if p_font == target_fontname and target_size_min <= p_size <= target_size_max:
            potential_labels.append(phrase)
        
        # Check if phrase is purely numeric
        if phrase.get('text', '').isdigit():
            potential_numbers.append(phrase)
            
    logger.info(f"Identified {len(potential_labels)} potential labels and {len(potential_numbers)} potential numbers based on format/content.")

    # Find number-label pairs based on proximity
    matched_numbers = set() # Avoid matching the same number multiple times if layouts are weird
    for label in potential_labels:
        best_match_num = None
        min_gap = max_h_gap + 1 # Initialize with value larger than tolerance
        
        for number in potential_numbers:
            if number.get('text') in matched_numbers:
                continue # Skip already matched numbers
                
            # Check vertical alignment
            v_aligned = abs(label.get('top', 0) - number.get('top', 0)) <= max_v_align_diff
            
            # Check horizontal proximity (number.x1 should be close to label.x0)
            h_gap = label.get('x0', 0) - number.get('x1', 0)
            h_adjacent = 0 <= h_gap <= max_h_gap
            
            if v_aligned and h_adjacent:
                # If multiple numbers are adjacent, pick the closest one
                if h_gap < min_gap:
                    min_gap = h_gap
                    best_match_num = number

        # If a suitable adjacent number was found, record the pair
        if best_match_num:
            numbered_labels.append({
                "label_number": best_match_num.get('text'),
                "label_text": label.get('text')
                # Optional: add coordinates if needed
                # "label_coords": {"x0": label["x0"], ...},
                # "number_coords": {"x0": best_match_num["x0"], ...}
            })
            matched_numbers.add(best_match_num.get('text')) # Mark number as used
            
    # Sort results potentially by number? (optional)
    numbered_labels.sort(key=lambda x: int(x["label_number"]) if x["label_number"].isdigit() else float('inf'))
            
    logger.info(f"Found {len(numbered_labels)} numbered labels.")
    return numbered_labels

def analyze_pdf_layout(pdf_path: str, target_page: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
    """Parses the PDF, extracts phrases, and identifies numbered labels.

    Args:
        pdf_path: Path to the PDF file.
        target_page: Optional 1-indexed page number to analyze. If None, analyze all.

    Returns:
        A dictionary where keys are 1-indexed page numbers and values are dictionaries
        containing page metadata and a list of identified numbered labels.
    """
    extracted_data_by_page = {}
    logger.info(f"Attempting to open PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Successfully opened PDF. Total Pages: {len(pdf.pages)}")
            
            pages_to_process = []
            if target_page:
                if 1 <= target_page <= len(pdf.pages):
                    pages_to_process = [(target_page, pdf.pages[target_page - 1])]
                    logger.info(f"Targeting page {target_page}.")
                else:
                    logger.error(f"Error: Page number {target_page} is out of range (1-{len(pdf.pages)}).")
                    return {}
            else:
                pages_to_process = list(enumerate(pdf.pages, 1))
                logger.info("Processing all pages.")

            for page_num, page in pages_to_process:
                logger.info(f"--- Processing Page {page_num} ---")
                
                # Extract words with formatting details
                words = page.extract_words(
                    x_tolerance=1, 
                    y_tolerance=1, 
                    keep_blank_chars=False, 
                    use_text_flow=True, 
                    horizontal_ltr=True,
                    extra_attrs=["fontname", "size", "stroking_color", "non_stroking_color"] # Crucial: Request extra attributes
                )
                
                logger.info(f"Found {len(words)} words on page {page_num}.")

                # Group words into phrases
                phrases = group_words_into_phrases(words)

                # --- Identify Numbered Labels --- #
                # Define target format based on the example "State income tax withheld"
                target_label_font = "YKIRGB+HelveticaNeueLTStd-Roman"
                target_label_size = 6.8317 
                size_tolerance = 0.2 # Allow slight size variations
                
                numbered_labels = extract_numbered_labels(
                    phrases,
                    target_fontname=target_label_font,
                    target_size_min=target_label_size - size_tolerance,
                    target_size_max=target_label_size + size_tolerance
                )
                # ---------------------------------

                # Output structure for the page
                page_output = {
                    "metadata": {
                        "page_number": page_num,
                        "width": page.width,
                        "height": page.height,
                    },
                    "numbered_labels": numbered_labels # Store the identified pairs
                    # Optionally keep raw phrases for debugging: "phrases": phrases 
                }
                
                extracted_data_by_page[page_num] = page_output

    except FileNotFoundError:
        logger.error(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during PDF parsing: {e}", exc_info=True)
        sys.exit(1)
    
    total_elements = sum(sum(len(v) for k, v in elements.items() if k != 'metadata') 
                         for elements in extracted_data_by_page.values())
    logger.info(f"Finished processing PDF. Extracted info for {total_elements} elements across {len(extracted_data_by_page)} page(s)." )
    return extracted_data_by_page

def main():
    """Main function to parse arguments and run the analysis."""
    args = parse_arguments()
    
    # Update log level based on arguments
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    # Also update root logger level if using basicConfig, or configure specific handlers
    logging.getLogger().setLevel(log_level) 
    
    logger.info(f"Starting PDF layout analysis for: {args.pdf_path}")
    # Pass the specific page number if provided
    extracted_layout_data = analyze_pdf_layout(args.pdf_path, target_page=args.page)
    
    # Output results
    if extracted_layout_data:
        # If a specific page was requested, print only that page's data
        pages_to_print = [args.page] if args.page else extracted_layout_data.keys()
        
        for page_num in pages_to_print:
            if page_num in extracted_layout_data:
                logger.info(f"--- Layout Data for Page {page_num} ---")
                # Convert Decimal objects to strings for JSON serialization if they appear
                print(json.dumps(extracted_layout_data[page_num], indent=2, default=str))
            else:
                # This case shouldn't happen with current logic, but good to have
                logger.warning(f"No data found for requested page {page_num}.")
    else:
        logger.info("No layout data was extracted.")

    logger.info("Script finished.")

if __name__ == "__main__":
    main() 