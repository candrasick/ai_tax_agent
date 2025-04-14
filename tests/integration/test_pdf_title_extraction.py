"""
Integration tests for PDF title extraction functionality.
"""

import pytest
import os
import pdfplumber
import logging

# Add project root to allow imports from ai_tax_agent
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_tax_agent.parsers.pdf_parser_utils import (
    extract_phrases_and_line_items,
    extract_form_schedule_titles
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

PDF_PATH = "data/tax_statistics/individuals.pdf"
START_PAGE = 3 # 1-based index

@pytest.mark.integration
def test_extract_titles_from_individuals_pdf():
    """Tests if titles can be extracted from each page of individuals.pdf (starting page 3)."""
    assert os.path.exists(PDF_PATH), f"Test PDF not found at {PDF_PATH}"

    failed_pages = []
    total_pages_processed = 0

    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            total_pages_in_pdf = len(pdf.pages)
            logging.info(f"Found {total_pages_in_pdf} pages in {os.path.basename(PDF_PATH)}. Starting test from page {START_PAGE}.")

            if START_PAGE > total_pages_in_pdf:
                 pytest.skip(f"Start page {START_PAGE} is beyond the total pages ({total_pages_in_pdf}) in the PDF.")

            # Iterate from START_PAGE (1-based) to the end
            for page_num in range(START_PAGE, total_pages_in_pdf + 1):
                logging.debug(f"Testing title extraction for page {page_num}...")
                page = pdf.pages[page_num - 1] # pdfplumber uses 0-based index
                total_pages_processed += 1

                # Extract header phrases using existing logic
                # Use default tolerances for this test, focus is on title extraction
                phrase_result = extract_phrases_and_line_items(page)
                header_phrases = phrase_result.get("header_phrases", [])
                
                if not header_phrases:
                    logging.warning(f"Page {page_num}: No header phrases extracted, cannot extract title.")
                    # Decide if this should be a failure or just a warning
                    # For now, let's count it as failure as title extraction needs headers
                    failed_pages.append(page_num)
                    continue 

                # Extract titles
                body_phrases = phrase_result.get("body_phrases", [])
                title_info = extract_form_schedule_titles(header_phrases, body_phrases)

                # Assert that at least one title was found
                form_title = title_info.get("form_title")
                schedule_title = title_info.get("schedule_title")
                
                title_found = bool(form_title or schedule_title)
                is_page_number = False
                if form_title:
                     # Check if the extracted form title is just the page number
                     if form_title.strip() == str(page_num):
                          is_page_number = True
                          logging.warning(f"Page {page_num}: Extracted form_title '{form_title}' matches the page number.")
                          

                if not title_found or is_page_number:
                    logging.warning(f"Page {page_num}: Failed to extract form_title or schedule_title. Found headers: {[h['text'] for h in header_phrases[:3]]}...")
                    failed_pages.append(page_num)

    except Exception as e:
        pytest.fail(f"An unexpected error occurred during PDF processing: {e}", pytrace=True)

    assert not failed_pages, (
        f"Failed to extract titles (form or schedule) for {len(failed_pages)}/{total_pages_processed} pages: {failed_pages}"
    )
    logging.info(f"Successfully validated title extraction for {total_pages_processed} pages (starting from page {START_PAGE}).") 