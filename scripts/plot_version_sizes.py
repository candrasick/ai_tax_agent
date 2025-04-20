#!/usr/bin/env python
"""Generates a plot showing the decrease in tax code size across versions."""

import logging
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

# Adjust import path based on your project structure
sys.path.append('.')

from ai_tax_agent.database.versioning import get_total_text_length_for_version
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import UsCodeSectionRevised

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_max_version() -> int:
    """Get the highest version number from the revised sections table."""
    db: Session = get_session()
    if not db:
        logger.error("Failed to get database session.")
        return 0
    
    try:
        max_version = db.query(UsCodeSectionRevised.version).order_by(UsCodeSectionRevised.version.desc()).first()
        return max_version[0] if max_version else 0
    finally:
        db.close()

def plot_version_sizes(output_path: Path):
    """Creates and saves a plot showing the decrease in tax code size across versions."""
    
    # Get the maximum version number
    max_version = get_max_version()
    logger.info(f"Found maximum version: {max_version}")
    
    # Collect data points
    versions = list(range(max_version + 1))  # Include version 0
    lengths = []
    
    # Get length for each version
    for version in versions:
        length = get_total_text_length_for_version(version)
        if length is None:
            logger.error(f"Failed to get length for version {version}")
            return
        lengths.append(length)
        logger.info(f"Version {version}: {length:,} characters")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot points and lines
    plt.plot(versions, lengths, 'bo-', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Version Number')
    plt.ylabel('Total Characters')
    plt.title('Tax Code Size Reduction Across Versions')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis with comma separators
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Add value labels on each point
    for i, length in enumerate(lengths):
        plt.annotate(f'{length:,}', 
                    (versions[i], length),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    # Calculate and display reduction percentages
    if lengths[0] > 0:  # Avoid division by zero
        for i in range(1, len(lengths)):
            reduction = ((lengths[0] - lengths[i]) / lengths[0]) * 100
            plt.annotate(f'-{reduction:.1f}%', 
                        (versions[i], lengths[i]),
                        textcoords="offset points",
                        xytext=(0,-20),
                        ha='center',
                        color='red')
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a plot showing tax code size reduction across versions.")
    parser.add_argument("--output", type=str, default="plots/version_sizes.png", 
                       help="Output path for the PNG file.")
    args = parser.parse_args()
    
    output_file = Path(args.output)
    plot_version_sizes(output_file) 