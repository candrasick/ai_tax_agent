import os
import re
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables (for DATABASE_URL)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logging.error("DATABASE_URL environment variable not set.")
    exit(1)

# Constants - Remove bulletin/XML paths, keep plots dir
PLOTS_DIR_DEFAULT = Path("plots/")

# Add constants for 4K resolution output
DPI = 300  # Standard high-quality DPI
WIDTH_PIXELS = 3840
HEIGHT_PIXELS = 2160
WIDTH_INCHES = WIDTH_PIXELS / DPI
HEIGHT_INCHES = HEIGHT_PIXELS / DPI

def set_high_res_style():
    """Configure plot style for 4K resolution."""
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': DPI,
        'savefig.dpi': DPI
    })

def get_bulletin_mention_counts_from_db(engine) -> pd.DataFrame:
    """Queries the database to count bulletin links per section, including titles."""
    logging.info("Querying database for bulletin mention counts and titles...")
    # Corrected query using schema diagram table/column names
    # Fetch section_title as well
    query = text("""
        SELECT ucs.section_number, ucs.section_title, COUNT(link.id) as mention_count
        FROM us_code_section ucs
        JOIN irs_bulletin_item_to_code_section link ON ucs.id = link.section_id
        GROUP BY ucs.section_number, ucs.section_title
        ORDER BY mention_count DESC;
    """)
    try:
        with engine.connect() as connection:
            result = connection.execute(query)
            # Return a DataFrame including the title
            df = pd.DataFrame(result.fetchall(), columns=['section', 'section_title', 'bulletin_mentions'])
            df.set_index('section', inplace=True)
            logging.info(f"Found bulletin mentions for {len(df)} sections in the database.")
            return df
    except Exception as e:
        # Check for specific "no such table" error
        if "no such table: us_code_section" in str(e).lower() or \
           "no such table: irs_bulletin_item_to_code_section" in str(e).lower():
            logging.error(f"Database error: Required table not found. Please ensure migrations and data population scripts have run. Details: {e}")
        else:
            logging.error(f"Database error querying bulletin mentions: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def get_amendment_counts_from_db(engine) -> pd.Series:
    """Queries the database for amendment counts per section number."""
    logging.info("Querying database for amendment counts...")
    # Corrected query using schema diagram table/column names
    query = text("""
        SELECT section_number, amendment_count 
        FROM us_code_section
        WHERE amendment_count IS NOT NULL AND amendment_count > 0;
    """)
    try:
        with engine.connect() as connection:
            result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=['section', 'amendment_counts'])
            if df.duplicated(subset=['section']).any():
                logging.warning("Duplicate section numbers found in amendment counts query. Aggregating counts.")
                df = df.groupby('section')['amendment_counts'].sum().reset_index()
            
            df.set_index('section', inplace=True)
            logging.info(f"Found amendment counts for {len(df)} sections in the database.")
            return df['amendment_counts']
    except Exception as e:
        if "no such column: amendment_count" in str(e).lower():
             logging.error(f"Database error: Column 'amendment_count' not found in 'us_code_section' table.")
             logging.error("Please ensure the script that populates this column (e.g., analyze_amendments.py) has been run.")
        elif "no such table: us_code_section" in str(e).lower():
             logging.error(f"Database error: Table 'us_code_section' not found. Please ensure migrations have run. Details: {e}")
        else:
            logging.error(f"Database error querying amendment counts: {e}")
        return pd.Series(dtype=int)

def plot_top_mentions(df: pd.DataFrame, output_file: Path):
    """Generates and saves a bar chart of top 10 sections by bulletin mentions, using titles as labels."""
    if df.empty or 'bulletin_mentions' not in df.columns or 'section_title' not in df.columns:
        logging.warning("DataFrame is empty or missing required columns ('bulletin_mentions', 'section_title'). Skipping top mentions plot.")
        return
    
    # Filter for sections with mentions and keep title
    df_mentions = df[df['bulletin_mentions'] > 0][['bulletin_mentions', 'section_title']].astype({'bulletin_mentions': int, 'section_title': str})
    if df_mentions.empty:
         logging.warning("No sections with bulletin mentions found. Skipping top mentions plot.")
         return

    logging.info(f"Generating Top 10 Bulletin Mentions plot with Titles at {WIDTH_PIXELS}x{HEIGHT_PIXELS} resolution...")
    # Sort by mentions, keep title associated with the index (section number)
    top_10 = df_mentions.nlargest(10, 'bulletin_mentions')

    if top_10.empty:
        logging.warning("No data available for top mentions plot.")
        return

    set_high_res_style()
    plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    
    # Plot using index for positioning, but we'll set labels from 'section_title'
    ax = sns.barplot(x=top_10.index, y=top_10['bulletin_mentions'], palette="viridis")
    
    # Set the x-axis tick labels to the section titles
    # Use short titles if available or truncate long ones for readability
    tick_labels = [f"{idx}: {title[:50]}{'...' if len(title)>50 else ''}" 
                   for idx, title in top_10['section_title'].items()]
    ax.set_xticklabels(tick_labels)
    
    plt.title('Top 10 Most Mentioned IRC Sections in Bulletins (from DB Links)')
    plt.xlabel('IRC Section (Number: Title)') # Updated axis label
    plt.ylabel('Number of Bulletin Links')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(top_10['bulletin_mentions']):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight')
        logging.info(f"Top mentions plot saved to {output_file}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save top mentions plot: {e}")


def plot_correlation_heatmap(df: pd.DataFrame, output_file: Path):
    """Generates and saves a correlation heatmap."""
    if df.empty or 'bulletin_mentions' not in df.columns or 'amendment_counts' not in df.columns:
        logging.warning("DataFrame is empty or missing required columns. Skipping correlation heatmap.")
        return
    
    df_numeric = df[['bulletin_mentions', 'amendment_counts']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df_numeric = df_numeric[(df_numeric['bulletin_mentions'] > 0) | (df_numeric['amendment_counts'] > 0)]

    if df_numeric.empty or df_numeric.shape[0] < 2:
         logging.warning("Not enough data with non-zero counts for correlation calculation. Skipping heatmap.")
         return

    logging.info(f"Generating Correlation Heatmap at {WIDTH_PIXELS}x{HEIGHT_PIXELS} resolution...")
    correlation_matrix = df_numeric.corr()

    set_high_res_style()
    plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    
    # Increase annotation size for 4K
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=.5,
                annot_kws={'size': 16})
                
    plt.title('Correlation between Bulletin Links and Amendment Counts (from DB)')
    plt.tight_layout()

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight')
        logging.info(f"Correlation heatmap saved to {output_file}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save correlation heatmap: {e}")


def main(plots_dir: Path):
    """Main analysis function using database as source."""
    logging.info("Starting analysis using database...")
    
    try:
        engine = create_engine(DATABASE_URL)
    except Exception as e:
        logging.error(f"Failed to create database engine: {e}")
        return

    # 1. Get bulletin mentions & titles from DB
    df_bulletins = get_bulletin_mention_counts_from_db(engine)

    # 2. Get amendment counts from DB
    s_amendment_counts = get_amendment_counts_from_db(engine)
    df_amendments = s_amendment_counts.to_frame(name='amendment_counts') # Convert Series to DataFrame

    # 3. Combine data
    logging.info("Combining bulletin and amendment data from database...")
    if df_bulletins.empty and df_amendments.empty:
        logging.error("No data retrieved from database for either bulletin mentions or amendments.")
        return
        
    # Merge on index (section_number). Bulletin df now has titles.
    df_combined = pd.merge(df_bulletins, df_amendments, left_index=True, right_index=True, how='outer')
    
    # Fill NaN values with 0 for counts, and empty string for title if missing
    df_combined['bulletin_mentions'].fillna(0, inplace=True)
    df_combined['amendment_counts'].fillna(0, inplace=True)
    df_combined['section_title'].fillna('N/A', inplace=True) # Handle sections only in amendments
    
    # Ensure integer types for counts after fillna
    df_combined = df_combined.astype({'bulletin_mentions': int, 'amendment_counts': int})

    logging.info(f"Combined data has {len(df_combined)} sections.")

    # 4. Generate Plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_top_mentions(df_combined, plots_dir / "top_bulletin_mentions_from_db.png")
    plot_correlation_heatmap(df_combined, plots_dir / "mentions_amendments_correlation_from_db.png")

    logging.info("Analysis script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze IRC section mentions (bulletin links) and amendments using database data.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR_DEFAULT,
        help=f"Directory to save output plots (default: {PLOTS_DIR_DEFAULT})"
    )

    args = parser.parse_args()

    # No need for file/dir validation here anymore

    main(args.plots_dir) 