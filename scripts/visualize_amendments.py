#!/usr/bin/env python
"""
Generates visualizations based on the amendment counts in the tax code sections.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project components
from ai_tax_agent.settings import settings # For DATABASE_URL

# --- Configuration ---
PLOTS_DIR = "plots"
TOP_N = 10
DB_URL = settings.database_url

def create_visualizations(output_dir: str = PLOTS_DIR):
    """Fetches data and creates amendment count visualizations."""
    # Set up high resolution output parameters
    DPI = 300  # Standard high-quality DPI
    WIDTH_PIXELS = 3840
    HEIGHT_PIXELS = 2160
    WIDTH_INCHES = WIDTH_PIXELS / DPI
    HEIGHT_INCHES = HEIGHT_PIXELS / DPI

    print(f"Connecting to database: {DB_URL}")
    try:
        engine = create_engine(DB_URL)
        # Fetch data: id, section_number, section_title, amendment_count
        query = "SELECT id, section_number, section_title, amendment_count FROM us_code_section ORDER BY amendment_count DESC"
        df = pd.read_sql(query, engine)
        print(f"Fetched {len(df)} sections from the database.")
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return

    if df.empty or 'amendment_count' not in df.columns:
        print("No data or missing 'amendment_count' column found. Cannot generate plots.")
        return

    # Ensure amendment_count is numeric
    df['amendment_count'] = pd.to_numeric(df['amendment_count'], errors='coerce')
    df = df.dropna(subset=['amendment_count'])
    df['amendment_count'] = df['amendment_count'].astype(int)

    # --- Calculate and Print Total Amendments ---
    total_amendments = df['amendment_count'].sum()
    print(f"\n>>> Total estimated amendments across all {len(df)} sections: {total_amendments}\n")
    # -------------------------------------------

    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Top N Most Amended Sections --- 
    try:
        print(f"Generating Top {TOP_N} Amended Sections plot at {WIDTH_PIXELS}x{HEIGHT_PIXELS} resolution...")
        top_n_df = df.nlargest(TOP_N, 'amendment_count')

        plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DPI)
        # Create labels like "§Number: Title"
        # Shorten long titles for better display
        max_title_len = 40
        top_n_df['label'] = top_n_df.apply(
            lambda row: f"§{row['section_number']}: {row['section_title'][:max_title_len] + '...' if row['section_title'] and len(row['section_title']) > max_title_len else row['section_title'] or '[No Title]'}", 
            axis=1
        )
        
        # Increase font sizes for 4K resolution
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14
        })
        
        # Use seaborn for potentially nicer aesthetics
        barplot = sns.barplot(x='amendment_count', y='label', data=top_n_df, palette="viridis", hue='label', dodge=False, legend=False)
        
        plt.title(f'Top {TOP_N} Most Amended U.S. Code Sections (Title 26)')
        plt.xlabel('Estimated Number of Amendments')
        plt.ylabel('Section')
        plt.tight_layout()
        
        # Add counts at the end of the bars with larger font
        for container in barplot.containers:
            barplot.bar_label(container, fmt='%d', padding=3, size=14)

        plot1_filename = os.path.join(output_dir, "top_amended_sections.png")
        plt.savefig(plot1_filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {plot1_filename}")
    except Exception as e:
        print(f"Error generating plot 1: {e}")

    # --- Plot 2: Distribution of Amendment Counts (Histogram) ---
    try:
        print(f"Generating Amendment Count Distribution plot at {WIDTH_PIXELS}x{HEIGHT_PIXELS} resolution...")
        plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DPI)

        # Use seaborn's histplot for flexibility
        # Filter out sections with 0 amendments for potentially better visualization of amended sections
        amended_sections = df[df['amendment_count'] > 0]['amendment_count']
        
        if not amended_sections.empty:
            # Determine appropriate bins, potentially log scale if distribution is highly skewed
            # Let seaborn choose bins initially, can adjust `bins` parameter if needed
            histplot = sns.histplot(amended_sections, kde=True, bins=50) # Added kde for smooth curve
            plt.title('Distribution of Amendment Counts Across All Sections (Count > 0)')
            plt.xlabel('Estimated Number of Amendments')
            plt.ylabel('Number of Sections')
            # Consider log scale for y-axis if counts vary widely
            # plt.yscale('log') 
            plt.tight_layout()
            
            plot2_filename = os.path.join(output_dir, "amendment_distribution.png")
            plt.savefig(plot2_filename, dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {plot2_filename}")
        else:
            print("No sections found with amendment_count > 0. Skipping distribution plot.")
            
    except Exception as e:
        print(f"Error generating plot 2: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for tax code amendment data.")
    parser.add_argument("--output-dir", type=str, default=PLOTS_DIR,
                        help=f"Directory to save the plots (default: {PLOTS_DIR})")
    
    args = parser.parse_args()
    create_visualizations(output_dir=args.output_dir) 