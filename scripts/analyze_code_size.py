import os
import re
import argparse
import datetime
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from tqdm import tqdm

def get_pdf_page_count(pdf_path):
    """Counts the number of pages in a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return 0
    except PdfReadError as e:
        print(f"Error reading PDF file {pdf_path}: {e}. Skipping file.")
        return 0
    except Exception as e:
        print(f"Unexpected error processing PDF {pdf_path}: {e}. Skipping file.")
        return 0

def analyze_code_size(tax_code_pdf, bulletins_dir, output_plot_file, projection_end_year=2050):
    """Analyzes tax code and bulletin sizes and generates a plot."""

    # 1. Count pages in the base tax code PDF
    print(f"Analyzing base tax code file: {tax_code_pdf}")
    base_code_pages = get_pdf_page_count(tax_code_pdf)
    if base_code_pages == 0:
        print("Warning: Base tax code PDF has 0 pages or could not be read.")
    else:
        print(f"Base tax code pages: {base_code_pages}")

    # 2. Count pages in bulletin PDFs per year
    print(f"Analyzing bulletins in directory: {bulletins_dir}")
    bulletin_pages_by_year = defaultdict(int)
    year_regex = re.compile(r"irb(\d{2})-\d+\.pdf", re.IGNORECASE)
    current_year = datetime.datetime.now().year

    if not os.path.isdir(bulletins_dir):
        print(f"Error: Bulletins directory not found at {bulletins_dir}")
        return

    # Get list of potential PDF files first
    pdf_filenames = [f for f in os.listdir(bulletins_dir) if f.lower().endswith('.pdf')]
    total_pdfs_to_process = len(pdf_filenames)
    print(f"Found {total_pdfs_to_process} PDF files to process.")

    total_bulletins_matched_and_processed = 0

    # Wrap the iteration with tqdm for progress bar
    for filename in tqdm(pdf_filenames, desc="Processing bulletins", unit="file"):
        match = year_regex.match(filename)
        if match:
            year_short = int(match.group(1))
            # Convert 2-digit year to 4-digit year (crude but likely effective for 1995+)
            year = 1900 + year_short if year_short >= 95 else 2000 + year_short

            # Exclude current year from aggregation for stable average calculation
            if year < current_year:
                pdf_path = os.path.join(bulletins_dir, filename)
                pages = get_pdf_page_count(pdf_path)
                bulletin_pages_by_year[year] += pages
                total_bulletins_matched_and_processed += 1
        # else: # Optional: Log skipped files if needed, but tqdm shows total files attempted
        #     # print(f"Warning: Skipping file with non-matching name: {filename}")
        pass # File name didn't match the irbYY-NN.pdf pattern

    print(f"Attempted processing {total_pdfs_to_process} files.")
    print(f"Matched pattern and aggregated {total_bulletins_matched_and_processed} bulletin PDFs (up to year {current_year - 1}).")

    if not bulletin_pages_by_year:
        print("Error: No bulletin pages found or processed. Cannot generate plot.")
        return

    # 3. Process data using Pandas
    df = pd.DataFrame.from_dict(bulletin_pages_by_year, orient='index', columns=['bulletin_pages'])
    df.index.name = 'year'
    df = df.sort_index()

    # Ensure all years from 1995 up to the last year found are present (fill missing with 0)
    start_year = 1995
    end_year_data = df.index.max()
    all_years_index = pd.RangeIndex(start=start_year, stop=end_year_data + 1, name='year')
    df = df.reindex(all_years_index, fill_value=0)

    # Calculate cumulative bulletin pages
    df['cumulative_bulletin_pages'] = df['bulletin_pages'].cumsum()

    # Calculate total pages (Base Code + Cumulative Bulletins)
    df['total_pages'] = base_code_pages + df['cumulative_bulletin_pages']

    # Calculate running average of *new* bulletin pages per year (for projection)
    # Use data only up to the year before the current year for stability
    valid_years_for_avg = df[df.index < current_year]
    if len(valid_years_for_avg['bulletin_pages']) > 0:
         # Calculate expanding mean directly on the yearly pages
        running_avg_series = valid_years_for_avg['bulletin_pages'].expanding().mean()
        # Get the last calculated average (most recent stable average)
        annual_page_growth_avg = running_avg_series.iloc[-1] if not running_avg_series.empty else 0
    else:
        annual_page_growth_avg = 0

    print(f"Calculated average annual bulletin page growth (up to {current_year - 1}): {annual_page_growth_avg:.2f} pages/year")

    # 4. Project future growth
    last_historical_year = df.index.max()
    last_historical_pages = df['total_pages'].iloc[-1]

    projection_years = list(range(last_historical_year + 1, projection_end_year + 1))
    projected_pages = []
    current_projected_pages = last_historical_pages
    for _ in projection_years:
        current_projected_pages += annual_page_growth_avg
        projected_pages.append(current_projected_pages)

    projection_df = pd.DataFrame({
        'year': projection_years,
        'projected_total_pages': projected_pages
    }).set_index('year')

    # 5. Plotting
    print(f"Generating plot: {output_plot_file}")
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot historical data
    ax.plot(df.index, df['total_pages'], label='Historical Total Pages (Code + Bulletins)', marker='.', linestyle='-')

    # Plot projected data
    ax.plot(projection_df.index, projection_df['projected_total_pages'], label=f'Projected Growth ({annual_page_growth_avg:.0f} pages/year avg)', linestyle='--')

    # Formatting
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Pages")
    ax.set_title("Estimated Growth of U.S. Tax Code + IRS Bulletins (Pages)")
    ax.legend()
    ax.grid(True)
    ax.ticklabel_format(style='plain', axis='y') # Avoid scientific notation on y-axis

    # Ensure plot directory exists
    os.makedirs(os.path.dirname(output_plot_file) or '.', exist_ok=True)
    try:
        plt.savefig(output_plot_file, dpi=300)
        print(f"Plot saved successfully to {output_plot_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Uncomment to display plot interactively if needed

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze page counts of tax code and bulletins and plot growth.")
    parser.add_argument(
        "--tax-code-pdf",
        default="data/usc26@119-4.pdf",
        help="Path to the main tax code PDF file."
    )
    parser.add_argument(
        "--bulletins-dir",
        default="data/bulletins",
        help="Directory containing the downloaded IRS bulletin PDFs."
    )
    parser.add_argument(
        "--output-plot",
        default="plots/tax_code_growth.png",
        help="Path to save the generated PNG plot file."
    )
    parser.add_argument(
        "--projection-end-year",
        type=int,
        default=2050,
        help="The final year for the growth projection."
    )

    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    output_dir = os.path.dirname(args.output_plot)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    analyze_code_size(
        tax_code_pdf=args.tax_code_pdf,
        bulletins_dir=args.bulletins_dir,
        output_plot_file=args.output_plot,
        projection_end_year=args.projection_end_year
    ) 