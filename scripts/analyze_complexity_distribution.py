#!/usr/bin/env python
"""Analyzes and plots the distribution of complexity scores across tax code sections."""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
import numpy as np
from scipy.stats import gaussian_kde

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up high resolution output parameters
DPI = 300
WIDTH_PX = 3840
HEIGHT_PX = 2160
WIDTH_IN = WIDTH_PX / DPI
HEIGHT_IN = HEIGHT_PX / DPI

def set_high_res_style():
    """Configure plot style for 4K resolution."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': DPI,
        'savefig.dpi': DPI
    })

def analyze_complexity_distribution(output_dir: Path):
    """Generate a 4K resolution plot of complexity score distribution."""
    logger.info("Fetching complexity scores from database...")
    
    # Create database engine
    engine = sa.create_engine(os.getenv("DATABASE_URL"))
    
    # Query complexity scores
    query = """
    SELECT complexity_score 
    FROM section_complexity 
    WHERE complexity_score IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.error("No complexity scores found in database")
        return
        
    logger.info(f"Retrieved {len(df)} complexity scores")
    
    # Create figure with 4K dimensions
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    
    # Create histogram with KDE
    scores = df['complexity_score']
    bins = np.linspace(0, scores.max(), 50)
    
    # Plot histogram
    n, bins, patches = ax.hist(scores, bins=bins, density=True, alpha=0.6, 
                             color='skyblue', edgecolor='black', label='Histogram')
    
    # Calculate and plot KDE
    kde = gaussian_kde(scores)
    x_range = np.linspace(0, scores.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='Density Estimate')
    
    # Add mean and median lines
    mean_score = scores.mean()
    median_score = scores.median()
    ax.axvline(mean_score, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_score:.2f}')
    ax.axvline(median_score, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_score:.2f}')
    
    # Formatting
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Tax Code Section Complexity Scores')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plt.savefig(output_dir / 'section_complexity_distribution.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved at {output_dir}/section_complexity_distribution.png")
    logger.info(f"Statistics: Mean={mean_score:.2f}, Median={median_score:.2f}, "
               f"Min={scores.min():.2f}, Max={scores.max():.2f}")

if __name__ == '__main__':
    analyze_complexity_distribution(Path('plots')) 