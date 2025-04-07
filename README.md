# AI Tax Agent: Simplifying the U.S. Tax Code

## Project Goal

This project aims to build an iterative AI agent that analyzes, simplifies, and streamlines the U.S. tax code. It uses LangChain, ChromaDB (vector store), and a relational database (SQLite via SQLAlchemy/Alembic) to achieve this.

The agent will:

1.  Ingest and analyze the Internal Revenue Code (IRC), IRS bulletins/guidance, and IRS Statistics of Income.
2.  Assess tax code sections based on complexity, financial impact, and avoidance opportunities.
3.  Iteratively simplify the code by removing/combining sections and rewriting them clearly.
4.  Track each simplification step, aiming for a final target of ~10 pages each for personal and business taxation.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Install Poetry:**
    *   If using Homebrew (macOS): `brew install poetry`
    *   Otherwise, follow the [official installation instructions](https://python-poetry.org/docs/#installation).

2.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd ai-tax-agent
    ```

3.  **Install dependencies:**
    ```bash
    poetry install
    ```

4.  **Set up the database:**
    The project uses SQLite and Alembic for database schema management. Run the initial migration:
    ```bash
    # Ensure the database file (tax_data.db) exists and is up-to-date
    make db-migrate
    ```
    *Note: Subsequent model changes will require generating new migrations (`poetry run alembic revision --autogenerate -m "..."`) before running `make db-migrate` again.*

## Usage

(Instructions on how to run the agent/components will be added here later.)