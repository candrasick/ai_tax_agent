import pdfplumber
import argparse
import sys
import os
from collections import defaultdict
import math
import logging
import re
import json

# Tolerance for floating point comparisons if needed
# FONT_SIZE_TOLERANCE = 0.01 
# COLOR_TOLERANCE = 0.01

# Add project root to Python path if necessary (adjust as needed)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# --- Add project root to sys.path to allow importing ai_tax_agent --- 
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from ai_tax_agent.parsers.pdf_parser_utils import (
    extract_amounts_by_color,
    extract_form_schedule_titles,
)

def display_page(mcid_groups, page_number, items_per_page=5):
    """Displays MCID groups for a specific page with pagination."""
    mcids = sorted(mcid_groups.keys())
    total_items = len(mcids)
    total_pages = math.ceil(total_items / items_per_page)
    current_page = 1

    while True:
        start_index = (current_page - 1) * items_per_page
        end_index = start_index + items_per_page
        items_to_display = mcids[start_index:end_index]

        print("-" * 80)
        print(f"PDF Page: {page_number + 1} | MCID Groups Page: {current_page}/{total_pages}")
        print(f"Displaying MCIDs {start_index + 1} to {min(end_index, total_items)} of {total_items}")
        print("-" * 80)

        if not items_to_display:
            print("No MCIDs found on this page.")
            break

        for mcid in items_to_display:
            group = mcid_groups[mcid]
            # Calculate approximate bounding box for the group
            min_x0 = min(c['x0'] for c in group['chars'])
            min_y0 = min(c['top'] for c in group['chars'])
            max_x1 = max(c['x1'] for c in group['chars'])
            max_y1 = max(c['bottom'] for c in group['chars'])
            bbox = (min_x0, min_y0, max_x1, max_y1)

            print(f"\nMCID: {mcid}")
            print(f"  Approx BBox: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})")
            # Print each character's text on a new line, indented
            # Note: This might be very verbose for large groups. Consider joining logical words/lines later if needed.
            full_text_builder = []
            for char_info in group['chars']:
                # Directly print each char - might be too granular
                # print(f"    '{char_info['text']}'") 
                # Build the full text first to print as one block for better readability
                full_text_builder.append(char_info['text'])
            
            # Print the reconstructed text, preserving original spacing as much as possible from char data
            # This simple join might lose some formatting nuance compared to original PDF structure
            reconstructed_text = "".join(full_text_builder)

            # --- Preprocess text: Add spaces --- 
            # Only preprocess needed: Add space before number pattern if not preceded by whitespace
            modified_text = re.sub(r'(?<=\S)(\d+[a-zA-Z]?)', r' \1', reconstructed_text)

            # --- Step 1: Print Modified Reconstructed Text --- 
            print("  Step 1: Modified Reconstructed Text:")
            if reconstructed_text:
                 # Print indented
                for line in modified_text.splitlines(): # Print the modified text
                    print(f"    {line}")
                # If reconstructed text is empty or has no newlines, print it directly
                if not modified_text.splitlines():
                    print(f"    {modified_text}")
            else:
                print("    [No text content extracted for this MCID group]")
            print("-" * 40) # Separator

            # Extract lines matching the pattern: <integer><optional letter><space><description><$> 
            # Final Pattern: Start, optional space, number/letter, space+, description, space+, literal $
            pattern = re.compile(r"^\s*(\d+[a-zA-Z]?)\s+(.*?)\s+\$")
            extracted_items = [] # Store dictionaries here

            # --- Logic-Based Parsing (Using Next Number as Delimiter) --- 
            # Find all number patterns (like " 1 ", " 2a ") - ensures space before/after
            # Pattern looks for space, number/letter, space (captures number/letter)
            number_pattern = re.compile(r"\s(\d+[a-zA-Z]?)\s")

            number_matches = list(number_pattern.finditer(modified_text))

            # Iterate through numbers, using the next number's start as the end delimiter
            for i in range(len(number_matches) - 1): # Go up to second-to-last number
                num_match = number_matches[i]
                next_num_match = number_matches[i+1]

                form_number = num_match.group(1) # The captured number/letter
                num_end_pos = num_match.end() # Position after the number pattern match

                # End position is the start of the *next* number pattern match
                title_end_pos = next_num_match.start()

                # Extract text between current number end and next number start
                title = modified_text[num_end_pos:title_end_pos].strip()

                # Basic filtering: avoid titles that are just numbers or very short
                if title and not title.isdigit() and len(title) > 1:
                    extracted_items.append({"form_number": form_number, "title": title})

            # --- Step 2: Print Extracted Items as JSON --- 
            print("  Extracted Form Items (JSON):")
            if extracted_items:
                # Use json.dumps for proper JSON formatting with indentation
                print(json.dumps(extracted_items, indent=4)) 
            else:
                print("  []") # Print empty JSON array if nothing found

            # Optionally print more details like fontname, size etc.
            # fonts = {c['fontname'] for c in group['chars']}
            # sizes = {c['size'] for c in group['chars']}
            # print(f"  Fonts: {fonts}")
            # print(f"  Sizes: {sizes}")

        print("-" * 80)
        if current_page >= total_pages:
            print("End of MCID groups for this PDF page.")
            break

        try:
            action = input("Press Space then Enter to continue, or Q then Enter to quit page: ").strip().lower()
            if action == 'q':
                break
            elif action == '': # Treat space + enter as empty string input
                current_page += 1
            else:
                print("Invalid input. Press Space then Enter to continue or Q then Enter to quit.")
        except EOFError: # Handle case where input stream is closed
             print("\nInput stream closed. Exiting pagination.")
             break


def analyze_pdf_fonts(pdf_path, page_num_one_indexed):
    """Analyzes a specific page of a PDF, separating amounts by color and grouping other text by font properties, including positions."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    page_index = page_num_one_indexed - 1 # pdfplumber uses 0-based index

    try:
        # Suppress pdfminer warnings (like CropBox missing)
        logging.getLogger("pdfminer").setLevel(logging.ERROR)

        with pdfplumber.open(pdf_path) as pdf:
            if page_index < 0 or page_index >= len(pdf.pages):
                print(f"Error: Page number {page_num_one_indexed} is out of range (1-{len(pdf.pages)}).", file=sys.stderr)
                sys.exit(1)

            page = pdf.pages[page_index]
            print(f"Analyzing PDF: '{os.path.basename(pdf_path)}', Page: {page_num_one_indexed}")

            # Group characters by font properties
            AMOUNT_COLOR_TUPLE = (0, 0, 0.5) # Target color for amounts
            MIN_HEADER_FONT_SIZE = 7.0 # Minimum size to be considered a header

            header_font_groups = defaultdict(list)
            body_font_groups = defaultdict(list)

            # --- Extract Amount Words using Word Extraction --- 
            amount_list = extract_amounts_by_color(page, AMOUNT_COLOR_TUPLE)

            # --- Group Remaining Text Characters --- 
            for char in page.chars:
                char_color = char.get('non_stroking_color')
                is_amount_char = False
                if isinstance(char_color, tuple) and len(char_color) >= 3:
                    if (abs(char_color[0] - AMOUNT_COLOR_TUPLE[0]) < 0.01 and
                        abs(char_color[1] - AMOUNT_COLOR_TUPLE[1]) < 0.01 and
                        abs(char_color[2] - AMOUNT_COLOR_TUPLE[2]) < 0.01):
                        is_amount_char = True

                # Only process characters not identified as part of an amount
                if not is_amount_char:
                    font = char.get('fontname')
                    size = char.get('size')
                    color = char_color
                    rounded_size = round(size, 1) if size is not None else None

                    # Use rounded size in the key
                    group_key = (font, rounded_size, color)

                    # Separate into header/body based on font size
                    if rounded_size is not None and rounded_size >= MIN_HEADER_FONT_SIZE:
                        header_font_groups[group_key].append(char)
                    else:
                        body_font_groups[group_key].append(char)

            if not header_font_groups and not body_font_groups and not amount_list:
                print(f"No characters found on page {page_num_one_indexed}.")
                return

            # Prepare JSON output
            header_elements_list = []
            body_elements_list = []

            # Sort groups for consistent output (optional, sorting by font name then size)
            sorted_header_keys = sorted(header_font_groups.keys(), key=lambda k: (k[0] or "", k[1] or 0))

            for key in sorted_header_keys:
                chars_in_group = header_font_groups[key]
                font, rounded_size, color = key
                text = "".join(c['text'] for c in chars_in_group)

                # Format color nicely for JSON
                color_str = repr(color) # Simple string representation

                # Calculate bounding box for the group
                if chars_in_group:
                    min_x0 = min(c['x0'] for c in chars_in_group)
                    min_y0 = min(c['top'] for c in chars_in_group)
                    max_x1 = max(c['x1'] for c in chars_in_group)
                    max_y1 = max(c['bottom'] for c in chars_in_group)
                    group_bbox = (min_x0, min_y0, max_x1, max_y1)
                else:
                    group_bbox = None # Should not happen if key exists

                header_elements_list.append({
                    "font": font,
                    "font_size": rounded_size, # Use rounded size
                    "font_color": color_str,
                    "text": text.strip(),
                    "position": group_bbox
                })

            # --- Process Body Groups --- 
            sorted_body_keys = sorted(body_font_groups.keys(), key=lambda k: (k[0] or "", k[1] or 0))

            for key in sorted_body_keys:
                chars_in_group = body_font_groups[key]
                font, rounded_size, color = key
                text = "".join(c['text'] for c in chars_in_group)

                # Format color nicely for JSON
                color_str = repr(color) # Simple string representation

                # Calculate bounding box for the group
                if chars_in_group:
                    min_x0 = min(c['x0'] for c in chars_in_group)
                    min_y0 = min(c['top'] for c in chars_in_group)
                    max_x1 = max(c['x1'] for c in chars_in_group)
                    max_y1 = max(c['bottom'] for c in chars_in_group)
                    group_bbox = (min_x0, min_y0, max_x1, max_y1)
                else:
                    group_bbox = None # Should not happen if key exists

                body_elements_list.append({
                    "font": font,
                    "font_size": rounded_size, # Use rounded size
                    "font_color": color_str,
                    "text": text.strip(),
                    "position": group_bbox
                })

            # --- Extract Form/Schedule Titles from Headers --- 
            title_info = extract_form_schedule_titles(header_elements_list)

            # Print the final JSON output
            final_output = {
                "form_title": title_info["form_title"],
                "schedule_title": title_info["schedule_title"],
                "amounts": amount_list,
                "body_elements": body_elements_list # Keep body elements
                # header_elements are not included in final output, only used for title extraction
            }
            print(json.dumps(final_output, indent=4))

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Analyze MCID groupings on a specific PDF page.')
    parser.add_argument('pdf_path', help='Path to the PDF file (e.g., data/tax_statistics/individuals.pdf)')
    parser.add_argument('page_number', type=int, help='The 1-based page number to analyze.')
    args = parser.parse_args()

    analyze_pdf_fonts(args.pdf_path, args.page_number)

if __name__ == "__main__":
    main()