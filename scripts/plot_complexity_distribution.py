#!/usr/bin/env python
"""Generates a 3D scatter plot of tax sections based on impact and complexity."""

import logging
import sys
import argparse
from pathlib import Path
import os
import sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
from sqlalchemy.orm import Session
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine, text

# Adjust import path based on your project structure
sys.path.append('.')

from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import UsCodeSection, SectionImpact, SectionComplexity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add constants for 4K resolution output
DPI = 300  # Standard high-quality DPI
WIDTH_PIXELS = 3840
HEIGHT_PIXELS = 2160
WIDTH_INCHES = WIDTH_PIXELS / DPI
HEIGHT_INCHES = HEIGHT_PIXELS / DPI

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
        'savefig.dpi': DPI,
        'lines.linewidth': 2.5,
        'grid.linewidth': 1.5
    })

def get_complexity_scores():
    """Retrieve complexity scores from the database."""
    try:
        # Create SQLAlchemy engine
        engine = create_engine('sqlite:///ai_tax_agent.db')
        
        with Session(engine) as session:
            # Use text() for raw SQL with a join to get complexity scores
            result = session.execute(text("""
                SELECT ucs.section_number, sc.complexity_score 
                FROM us_code_section ucs
                JOIN section_complexity sc ON ucs.id = sc.section_id
                WHERE sc.complexity_score IS NOT NULL
                ORDER BY sc.complexity_score
            """))
            
            scores = [float(row[1]) for row in result]
            logger.info(f"Retrieved {len(scores)} complexity scores from database")
            return scores
            
    except Exception as e:
        logger.error(f"Error retrieving complexity scores: {e}")
        return []

def plot_complexity_distribution(scores, output_path):
    """Generate a 4K resolution histogram with KDE curve of complexity scores."""
    set_high_res_style()
    
    # Create figure with 4K dimensions
    fig, ax = plt.subplots(figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    
    # Calculate histogram with more bins for smoother distribution
    bins = np.linspace(0, 10, 41)  # 40 bins from 0 to 10 for more granularity
    n, bins, patches = ax.hist(scores, bins=bins, alpha=0.6, color='lightsteelblue', 
                             edgecolor='black', linewidth=1)
    
    # Calculate and plot KDE
    kde = gaussian_kde(scores)
    x_range = np.linspace(0, 10, 500)  # More points for smoother curve
    kde_values = kde(x_range)
    
    # Scale KDE to match histogram height
    scaling_factor = np.max(n) / np.max(kde_values)
    ax.plot(x_range, kde_values * scaling_factor, color='blue', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Complexity Score', fontsize=16, labelpad=10)
    ax.set_ylabel('Number of Sections', fontsize=16, labelpad=10)
    ax.set_title('Distribution of Complexity Scores Across All Sections', 
                fontsize=18, pad=20)
    
    # Set axis limits with some padding
    ax.set_xlim(-0.2, 10.2)
    ymax = np.max(n) * 1.1  # Add 10% padding to y-axis
    ax.set_ylim(0, ymax)
    
    # Add grid with specific style
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add light gray background
    ax.set_facecolor('#f0f0f0')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Plot saved to {output_path} at {WIDTH_PIXELS}x{HEIGHT_PIXELS} resolution")

def fetch_plot_data(db: Session) -> pd.DataFrame:
    """Fetches data required for the 3D scatter plot from the database."""
    logger.info("Fetching data for 3D scatter plot...")
    try:
        query = (
            db.query(
                UsCodeSection.id,
                UsCodeSection.section_number,
                SectionImpact.revenue_impact,
                SectionImpact.entity_impact,
                SectionComplexity.complexity_score
            )
            .join(SectionImpact, UsCodeSection.id == SectionImpact.section_id)
            .join(SectionComplexity, UsCodeSection.id == SectionComplexity.section_id)
            # Ensure all required fields are not NULL
            .filter(
                SectionImpact.revenue_impact.isnot(None),
                SectionImpact.entity_impact.isnot(None),
                SectionComplexity.complexity_score.isnot(None)
            )
        )
        
        df = pd.read_sql(query.statement, db.bind)
        logger.info(f"Fetched {len(df)} sections with complete impact and complexity data.")
        return df
    except Exception as e:
        logger.error(f"Database error fetching plot data: {e}", exc_info=True)
        raise # Re-raise the exception to be caught in main

def draw_cuboid(ax, x_range, y_range, z_range, color='red', alpha=0.1):
    """Draws a semi-transparent cuboid on the 3D axes."""
    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_surface(xx, yy, np.full_like(xx, z_range[0]), color=color, alpha=alpha) # Bottom face
    ax.plot_surface(xx, yy, np.full_like(xx, z_range[1]), color=color, alpha=alpha) # Top face

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_surface(xx, np.full_like(xx, y_range[0]), zz, color=color, alpha=alpha) # Side face y=min
    ax.plot_surface(xx, np.full_like(xx, y_range[1]), zz, color=color, alpha=alpha) # Side face y=max

    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_surface(np.full_like(yy, x_range[0]), yy, zz, color=color, alpha=alpha) # Side face x=min
    ax.plot_surface(np.full_like(yy, x_range[1]), yy, zz, color=color, alpha=alpha) # Side face x=max


def create_3d_scatter_plot(df: pd.DataFrame, output_path: Path):
    """Creates and saves the 3D scatter plot with highlighted regions."""
    if df.empty:
        logger.warning("DataFrame is empty after processing. Cannot create plot.")
        return

    logger.info(f"Creating 3D scatter plot with {len(df)} data points...")
    # Calculate figure size based on desired pixel resolution and DPI
    width_inches = 3840 / DPI
    height_inches = 2160 / DPI
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=DPI)  # This will give us 3840x2160 pixels
    ax = fig.add_subplot(111, projection='3d')

    # --- Define Quantile Thresholds ---
    # Use processed log/normalized data for threshold definition
    low_impact_thresh = 0.33
    high_impact_thresh = 0.66
    low_complexity_thresh = 0.33
    high_complexity_thresh = 0.66

    rev_impact_q = df['log_revenue_impact'].quantile([low_impact_thresh, high_impact_thresh])
    ent_impact_q = df['log_entity_impact'].quantile([low_impact_thresh, high_impact_thresh])
    complexity_q = df['normalized_complexity'].quantile([low_complexity_thresh, high_complexity_thresh])

    # Min/Max values for ranges
    min_rev_impact, max_rev_impact = df['log_revenue_impact'].min(), df['log_revenue_impact'].max()
    min_ent_impact, max_ent_impact = df['log_entity_impact'].min(), df['log_entity_impact'].max()
    min_complexity, max_complexity = df['normalized_complexity'].min(), df['normalized_complexity'].max()
    # Add a small buffer to min/max for visual clarity if needed
    buffer_factor = 0.01
    min_rev_impact -= (max_rev_impact - min_rev_impact) * buffer_factor
    max_rev_impact += (max_rev_impact - min_rev_impact) * buffer_factor
    min_ent_impact -= (max_ent_impact - min_ent_impact) * buffer_factor
    max_ent_impact += (max_ent_impact - min_ent_impact) * buffer_factor
    min_complexity -= (max_complexity - min_complexity) * buffer_factor if max_complexity > min_complexity else 0.05 # Ensure non-zero range
    max_complexity += (max_complexity - min_complexity) * buffer_factor if max_complexity > min_complexity else 0.05 # Ensure non-zero range
    # Clamp complexity to [0, 1] if normalization worked as expected
    min_complexity = max(0.0, min_complexity)
    max_complexity = min(1.0, max_complexity)

    # --- Define Region Coordinates ---
    # Region 1: High impact (both), high complexity
    region1_x = [rev_impact_q[high_impact_thresh], max_rev_impact]
    region1_y = [ent_impact_q[high_impact_thresh], max_ent_impact]
    region1_z = [complexity_q[high_complexity_thresh], max_complexity]

    # Region 2: Low impact (both), low complexity
    region2_x = [min_rev_impact, rev_impact_q[low_impact_thresh]]
    region2_y = [min_ent_impact, ent_impact_q[low_impact_thresh]]
    region2_z = [min_complexity, complexity_q[low_complexity_thresh]]

    # Region 3: Low to Moderate impact (both), high complexity
    region3_x = [min_rev_impact, rev_impact_q[high_impact_thresh]]
    region3_y = [min_ent_impact, ent_impact_q[high_impact_thresh]]
    region3_z = [complexity_q[high_complexity_thresh], max_complexity]
    # ---

    # --- Draw Highlighted Regions ---
    region_alpha = 0.1 # Transparency level
    draw_cuboid(ax, region1_x, region1_y, region1_z, color='red', alpha=region_alpha)
    draw_cuboid(ax, region2_x, region2_y, region2_z, color='green', alpha=region_alpha)
    draw_cuboid(ax, region3_x, region3_y, region3_z, color='blue', alpha=region_alpha)
    # ---

    # Scatter plot using processed data
    scatter = ax.scatter(
        df['log_revenue_impact'], 
        df['log_entity_impact'], 
        df['normalized_complexity'], 
        c=df['normalized_complexity'], # Color by complexity
        cmap='viridis', # Colormap (options: 'plasma', 'inferno', 'magma', 'cividis', etc.)
        marker='o'
    )

    # Adding labels and title
    ax.set_xlabel('Financial Impact (Log10 Scale, $)')
    ax.set_ylabel('Entity Impact (Log10 Scale, People/Forms)')
    ax.set_zlabel('Complexity Score (Normalized 0-1)')
    plt.title('Tax Section Impact vs. Complexity')

    # Add a color bar
    fig.colorbar(scatter, label='Normalized Complexity Score')

    # --- Add Region Labels ---
    # Place text annotations near the center of each region's visible face
    ax.text(np.mean(region1_x), np.mean(region1_y), np.mean(region1_z), "High Impact/High Complexity", color='red', zorder=10)
    ax.text(np.mean(region2_x), np.mean(region2_y), np.mean(region2_z), "Low Impact/Low Complexity", color='green', zorder=10)
    ax.text(np.mean(region3_x), np.mean(region3_y), np.mean(region3_z), "Low to Moderate Impact/High Complexity", color='blue', zorder=10)
    # ---

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    try:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_path} at {3840}x{2160} resolution")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3D scatter plot of tax section impact vs complexity.")
    parser.add_argument("--output", type=str, default="plots/section_impact_complexity_3d.png", help="Output path for the PNG file.")
    # Add other arguments if needed later (e.g., filters)
    args = parser.parse_args()

    output_file = Path(args.output)

    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session. Exiting.")
        sys.exit(1)

    try:
        plot_df = fetch_plot_data(db)

        if plot_df.empty:
             logger.warning("No data fetched from database with required fields. Exiting.")
             sys.exit(0)

        # --- Data Preprocessing ---
        # Convert impacts to numeric, coercing errors (though query should filter nulls)
        plot_df['revenue_impact'] = pd.to_numeric(plot_df['revenue_impact'], errors='coerce')
        plot_df['entity_impact'] = pd.to_numeric(plot_df['entity_impact'], errors='coerce')
        plot_df['complexity_score'] = pd.to_numeric(plot_df['complexity_score'], errors='coerce')
        
        # Drop rows where conversion failed (if any)
        plot_df.dropna(subset=['revenue_impact', 'entity_impact', 'complexity_score'], inplace=True)

        # Filter out non-positive values for log scale
        initial_count = len(plot_df)
        plot_df = plot_df[(plot_df['revenue_impact'] > 0) & (plot_df['entity_impact'] > 0)]
        filtered_count = initial_count - len(plot_df)
        if filtered_count > 0:
             logger.info(f"Filtered out {filtered_count} sections with non-positive impact values.")
             
        if plot_df.empty:
             logger.warning("No data remaining after filtering for log scale. Exiting.")
             sys.exit(0)

        # Log Scale
        plot_df['log_revenue_impact'] = np.log10(plot_df['revenue_impact'])
        plot_df['log_entity_impact'] = np.log10(plot_df['entity_impact'])

        # Normalize Complexity Score (Min-Max Scaling 0-1)
        min_complexity = plot_df['complexity_score'].min()
        max_complexity = plot_df['complexity_score'].max()
        if max_complexity > min_complexity:
            plot_df['normalized_complexity'] = (plot_df['complexity_score'] - min_complexity) / (max_complexity - min_complexity)
        else:
             # Handle case where all complexities are the same
             logger.warning("All complexity scores are identical. Setting normalized score to 0.5.")
             plot_df['normalized_complexity'] = 0.5 
        # -------------------------

        create_3d_scatter_plot(plot_df, output_file)

    except Exception as e:
        logger.error(f"An error occurred during script execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if db:
            db.close()
            logger.debug("Database session closed.")

    # Generate and save complexity distribution plot
    output_path = "plots/complexity_distribution.png"
    scores = get_complexity_scores()
    
    if scores:
        logger.info(f"Generating 4K complexity distribution plot with {len(scores)} sections...")
        plot_complexity_distribution(scores, output_path)
    else:
        logger.error("No complexity scores found in database") 