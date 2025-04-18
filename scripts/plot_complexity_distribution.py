#!/usr/bin/env python
"""Generates a 3D scatter plot of tax sections based on impact and complexity."""

import logging
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
from sqlalchemy.orm import Session

# Adjust import path based on your project structure
sys.path.append('.')

from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import UsCodeSection, SectionImpact, SectionComplexity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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

def create_3d_scatter_plot(df: pd.DataFrame, output_path: Path):
    """Creates and saves the 3D scatter plot."""
    if df.empty:
        logger.warning("DataFrame is empty after processing. Cannot create plot.")
        return

    logger.info(f"Creating 3D scatter plot with {len(df)} data points...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

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

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_path}")
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