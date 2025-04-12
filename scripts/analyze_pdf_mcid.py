import pdfplumber
import argparse
import sys
import os
from collections import defaultdict
import math
import logging
import re
import json

# Add project root to Python path if necessary (adjust as needed)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

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
            # Add space *before* '$'
            modified_text = re.sub(r'\$', r' $', reconstructed_text)
            # Add space before number pattern (like '1', '2a') if not preceded by whitespace
            modified_text = re.sub(r'(?<=\S)(\d+[a-zA-Z]?)', r' \1', modified_text)

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
            for line in modified_text.splitlines(): # Use modified text for regex matching
                match = pattern.search(line.strip()) # Strip line whitespace before matching
                if match:
                    form_number = match.group(1).strip()
                    title = match.group(2).strip()
                    extracted_items.append({"form_number": form_number, "title": title})

            # --- Logic-Based Parsing --- 
            # Find all number patterns (like " 1 ", " 2a ")
            # Pattern looks for space, number/letter, space (captures number/letter)
            number_pattern = re.compile(r"\s(\d+[a-zA-Z]?)\s")
            # Find all spaced dollar signs (" $")
            dollar_pattern = re.compile(r"\s\$")

            number_matches = list(number_pattern.finditer(modified_text))
            dollar_matches = list(dollar_pattern.finditer(modified_text))

            dollar_positions = [match.start() for match in dollar_matches]
            dollar_idx = 0

            # Create a set of number start positions for quick lookup during validation
            # We need the start position of the actual number, not the preceding space
            number_start_positions = {match.start(1) for match in number_matches}

            for i, num_match in enumerate(number_matches):
                form_number = num_match.group(1) # The captured number/letter
                num_end_pos = num_match.end() # Position after the number pattern match

                # Find the next dollar sign position after the current number
                next_dollar_pos = -1
                temp_dollar_idx = dollar_idx # Use a temporary index for searching
                while temp_dollar_idx < len(dollar_positions):
                    current_dollar_pos = dollar_positions[temp_dollar_idx]
                    if current_dollar_pos > num_end_pos:
                        # Found the first dollar sign *after* the current number ends
                        next_dollar_pos = current_dollar_pos
                        # Optimization: update main dollar_idx so next number search starts here
                        # This assumes numbers and dollars appear roughly in order
                        dollar_idx = temp_dollar_idx
                        break
                    temp_dollar_idx += 1

                if next_dollar_pos != -1:
                    # --- Validation Step --- 
                    # Check if another number pattern starts within the potential title text
                    potential_title_slice = modified_text[num_end_pos:next_dollar_pos]
                    contains_another_number = False
                    # Search for the start of another number pattern within the slice
                    intervening_match = number_pattern.search(potential_title_slice)
                    if intervening_match:
                        contains_another_number = True
                        # Optional: Log skipped matches for debugging
                        # logger.debug(f"Skipping potential match for '{form_number}': Found intervening number pattern near '{intervening_match.group(1)}' in title slice: '{potential_title_slice[:50]}...' ")

                    # Only proceed if no other number pattern was found in the slice
                    if not contains_another_number:
                        # Extract text between number match end and dollar match start
                        title = potential_title_slice.strip()
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


def analyze_pdf_mcids(pdf_path, page_num_one_indexed):
    """Analyzes a specific page of a PDF, grouping text by MCID."""
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

            mcid_groups = defaultdict(lambda: {"chars": []})

            # Extract characters and group by mcid
            chars = page.chars
            for char in chars:
                mcid = char.get('mcid') # Use .get() for safety, default is None
                if mcid is not None: # Only group characters that have an MCID
                     mcid_groups[mcid]["chars"].append(char)
                # else:
                    # Handle characters without MCID if needed
                    # print(f"Char '{char['text']}' has no MCID")


            if not mcid_groups:
                 print(f"No characters with MCIDs found on page {page_num_one_indexed}.")
                 return

            # Display results with pagination
            display_page(mcid_groups, page_index) # Pass 0-based index

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Analyze MCID groupings on a specific PDF page.')
    parser.add_argument('pdf_path', help='Path to the PDF file (e.g., data/tax_statistics/individuals.pdf)')
    parser.add_argument('page_number', type=int, help='The 1-based page number to analyze.')
    args = parser.parse_args()

    analyze_pdf_mcids(args.pdf_path, args.page_number)

if __name__ == "__main__":
    main()