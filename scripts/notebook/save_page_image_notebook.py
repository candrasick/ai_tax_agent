#!/usr/bin/env python
"""
Notebook-friendly script to save a specific page of a PDF as an image.
"""

import os
import sys
import argparse
import logging
import pdfplumber

# Add project root to path (adjust if necessary)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure Pillow is installed: pip install Pillow

def save_pdf_page_as_image(pdf_path: str, page_num: int, output_image_path: str):
    """Opens a PDF, extracts a specific page, and saves it as a PNG image."""
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not (1 <= page_num <= len(pdf.pages)):
                logging.error(f"Page number {page_num} is out of range (1-{len(pdf.pages)})." )
                return False

            page = pdf.pages[page_num - 1] # pdfplumber is 0-indexed
            
            # Increase resolution for better quality for LLM analysis
            img = page.to_image(resolution=150) 
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_image_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created output directory: {output_dir}")
            
            img.save(output_image_path, format="PNG")
            logging.info(f"Successfully saved page {page_num} from '{os.path.basename(pdf_path)}' to {output_image_path}")
            return True
            
    except ImportError:
        logging.error("Pillow library not found. Please install it: pip install Pillow")
        return False
    except Exception as e:
        logging.error(f"An error occurred processing PDF page {page_num}: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Save a specific PDF page as a PNG image.")
    parser.add_argument("--pdf-path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--page-num", type=int, required=True, help="The 1-based page number to save.")
    parser.add_argument("--output-image", required=True, help="Path to save the output PNG image file.")
    args = parser.parse_args()

    save_pdf_page_as_image(args.pdf_path, args.page_num, args.output_image)

if __name__ == "__main__":
    main() 