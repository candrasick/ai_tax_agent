import requests
import os
import argparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import time
import datetime
import json # Need to parse JSON response

# AJAX endpoint for searching bulletins
AJAX_URL = "https://www.irs.gov/views/ajax"
# Base URL for resolving relative PDF links
IRS_BASE_URL = "https://www.irs.gov/"

def download_pdf(pdf_url, destination_folder):
    """Downloads a single PDF if it doesn't exist locally."""
    filename = pdf_url.split('/')[-1]
    # Ensure filename is valid (e.g., remove potential query params if any)
    filename = filename.split('?')[0]
    filepath = os.path.join(destination_folder, filename)

    if os.path.exists(filepath):
        print(f"Skipping existing file: {filename}")
        return True # Indicate skipped

    print(f"Downloading: {filename} from {pdf_url}")
    try:
        response = requests.get(pdf_url, stream=True, timeout=60) # Increased timeout for larger files
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {filename}")
        return True # Indicate success
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(filepath):
            os.remove(filepath)
        return False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def scrape_bulletins_for_year(year, destination_folder, delay=1):
    """Scrapes bulletins for a specific year using the AJAX search endpoint.

    Returns:
        bool: True if results were found (even if downloads failed), 
              False if 'no results' message was detected.
        None: On critical error during request/parsing.
    """
    print(f"--- Querying bulletins for year: {year} ---")

    # Parameters based on observed AJAX request
    # Some parameters like view_dom_id might change, but hopefully core ones are stable
    params = {
        '_wrapper_format': 'drupal_ajax',
        'find': str(year),
        'items_per_page': '200', # Get max results per page
        'view_name': 'pup_picklists',
        'view_display_id': 'internal_revenue_bulletins',
        'view_args': '',
        'view_path': '/node/108761',
        'view_base_path': '',
        #'view_dom_id': '8b24276a01df78738dbe7c79886e1f0fcb52a08ad7c6ba1d414cd6c9e2463b6c', # May be dynamic
        'pager_element': '0',
        '_drupal_ajax': '1',
        # 'ajax_page_state[theme]': 'pup_irs', # Less likely needed
        # 'ajax_page_state[theme_token]': '', # Less likely needed
        # 'ajax_page_state[libraries]': '...' # Less likely needed
    }

    try:
        response = requests.get(AJAX_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # The response is JSON containing commands, one of which inserts HTML
        ajax_commands = response.json()
        
        html_content = ""
        for command in ajax_commands:
            if command.get('command') == 'insert' and 'data' in command:
                html_content = command['data']
                break
        
        if not html_content:
            print(f"Warning: Could not find HTML data in AJAX response for year {year}.")
            # Decide how to handle this: maybe treat as no results or error?
            return False # Treat as no results found

        # Parse the extracted HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Check for the "no results" message
        no_results_text = "Your search did not return any results"
        if no_results_text in soup.get_text():
            print(f"Found 'no results' message for year {year}. Stopping.")
            return False # Signal no results found

        # Find PDF links within the extracted HTML
        # Links seem to be directly to the PDF like /pub/irs-irbs/irbYY-NN.pdf
        pdf_links_found = False
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf') and '/pub/irs-irbs/' in href.lower():
                pdf_links_found = True
                pdf_url = urljoin(IRS_BASE_URL, href)
                
                # Download the PDF (function handles skipping existing)
                time.sleep(delay) # Delay before each download attempt
                download_pdf(pdf_url, destination_folder)
                # We don't need to stop on download error, just continue
        
        if not pdf_links_found:
            print(f"Warning: Found results for {year}, but no PDF links matching pattern were extracted.")
            # This might indicate a change in HTML structure within the AJAX response

        return True # Signal that results were found (search should continue to previous year)

    except requests.exceptions.RequestException as e:
        print(f"Error querying year {year}: {e}")
        return None # Signal critical error
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for year {year}: {e}")
        print(f"Response text: {response.text[:500]}...") # Log part of the response
        return None # Signal critical error
    except Exception as e:
        print(f"An unexpected error occurred processing year {year}: {e}")
        return None # Signal critical error


def main():
    parser = argparse.ArgumentParser(description="Download IRS Internal Revenue Bulletin PDFs using year search.")
    parser.add_argument(
        "-d", "--destination",
        default="data/bulletins",
        help="Folder to save the downloaded PDF files (default: data/bulletins)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=datetime.datetime.now().year + 1, # Start slightly ahead for future bulletins
        help="The most recent year to start searching from."
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=1995, # IRS site mentions 1995 availability
        help="The oldest year to search down to."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between download requests (default: 1.0)"
    )
    args = parser.parse_args()

    destination_folder = args.destination
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Ensured destination folder exists: {destination_folder}")

    total_files_processed = 0 # Approximate count, not easily tracked now

    for year in range(args.start_year, args.end_year - 1, -1):
        result = scrape_bulletins_for_year(year, destination_folder, args.delay)
        
        if result is False:
            # Found "no results" message, normal termination
            break 
        elif result is None:
            # Critical error occurred during processing for this year
            print(f"Stopping due to critical error processing year {year}.")
            break
        # If result is True, continue to the next (older) year
        
        # Add a small delay between years if desired
        # time.sleep(1)

    print(f"--- Scraping Finished ---")
    # print(f"Total unique PDF filenames processed (attempted/skipped): {total_files_processed}") # Harder to track accurately now
    print(f"Files saved in: {destination_folder}")

if __name__ == "__main__":
    main() 