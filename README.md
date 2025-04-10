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

3.  **Install Git LFS:**
    This project uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to manage large data files (e.g., tax documents in the `data/` directory). You need to install the Git LFS client to download these files correctly.
    *   If using Homebrew (macOS): `brew install git-lfs`
    *   Otherwise, follow the [official installation instructions](https://git-lfs.github.com/).
    After installing, ensure LFS is initialized for your user account (usually only needed once per machine):
    ```bash
    git lfs install 
    ```
    If you cloned the repository *before* installing Git LFS, navigate into the repository directory and run `git lfs pull` to download the large files.

4.  **Install dependencies:**
    ```bash
    poetry install
    ```

5.  **Set up the database:**
    The project uses SQLite and Alembic for database schema management. Run the initial migration:
    ```bash
    # Ensure the database file (tax_data.db) exists and is up-to-date
    make db-migrate
    ```
    *Note: Subsequent model changes will require generating new migrations (`poetry run alembic revision --autogenerate -m "..."`) before running `make db-migrate` again.*

## Usage

### Makefile Targets

This project uses a `Makefile` to streamline common tasks. Here's a breakdown of the available targets and their recommended execution order:

**Recommended Execution Order (for initial setup & data processing):**

1.  `make download-bulletins`
2.  `make db-migrate`
3.  `make parse-tax-code` (Requires `data/usc26.xml`)
4.  `make parse-bulletins` (Requires downloaded PDFs in `data/irb/`)
5.  `make link-bulletins` (Links parsed bulletins to parsed code sections in the DB)
6.  (Optional Analysis) `make analyze-size`, `make analyze-mentions`
7.  (Testing) `make test`, `make test-unit`, `make test-integration`

**Target Descriptions:**

*   `download-bulletins`
    *   Runs `scripts/download_irb_bulletins.py` to download Internal Revenue Bulletins (IRBs) as PDFs into the `data/irb/` directory.
*   `db-migrate`
    *   Applies database migrations using Alembic. Creates or updates the database schema in `data/tax_data.db` based on models defined in `ai_tax_agent/db/models.py`.
*   `parse-tax-code`
    *   Runs `scripts/parse_tax_code.py` to parse the main IRC XML file (`data/usc26.xml`) and populate the `us_code_section` table in the database.
*   `parse-bulletins`
    *   Runs `scripts/parse_bulletins.py` to parse the downloaded bulletin PDFs, extracting metadata and potentially text, populating the `irs_bulletin` and `irs_bulletin_item` tables.
*   `link-bulletins`
    *   Runs `scripts/link_bulletins_to_sections.py` to identify references to IRC sections within the parsed bulletin text and create links in the `irs_bulletin_item_to_code_section` table.
*   `analyze-size`
    *   Runs `scripts/analyze_code_size.py` to count pages in the main tax code PDF and downloaded bulletins, generating a plot of size over time (`plots/tax_code_growth.png`).
*   `analyze-mentions`
    *   Runs `scripts/analyze_section_mentions.py` using data from the database to analyze the frequency of section mentions in bulletins and their correlation with amendment counts. Generates plots (`plots/top_bulletin_mentions_from_db.png`, `plots/mentions_amendments_correlation_from_db.png`).
*   `test`
    *   Runs all available tests (unit and integration).
*   `test-unit`
    *   Runs only the unit tests located in `tests/unit`.
*   `test-integration`
    *   Runs only the integration tests located in `tests/integration`.

*(Note: Some targets like `parse-tax-code-custom`, `analyze-amendments`, etc., might exist but are not fully described here. Refer to the `Makefile` for details.)*

### Database Schema (`data/tax_data.db`)

The project uses an SQLite database managed by SQLAlchemy and Alembic. The core tables are:

*   `alembic_version`:
    *   Tracks the current database migration version (used by Alembic).
*   `us_code_section`:
    *   Stores parsed sections of the U.S. Internal Revenue Code (Title 26).
    *   Key columns: `id`, `section_number`, `section_title`, `full_text`, `amendment_count`, `amendments_text`, `core_text`.
*   `irs_bulletin`:
    *   Stores metadata about each downloaded Internal Revenue Bulletin.
    *   Key columns: `id`, `bulletin_number`, `bulletin_date`, `source_url`.
*   `irs_bulletin_item`:
    *   Stores individual items (like Revenue Rulings, Notices, etc.) extracted from within each bulletin.
    *   Key columns: `id`, `bulletin_id` (FK to `irs_bulletin`), `item_type`, `item_number`, `title`, `full_text`, `referenced_sections`.
*   `irs_bulletin_item_to_code_section`:
    *   A many-to-many link table connecting bulletin items (`bulletin_item_id`) to the specific code sections (`section_id` -> `us_code_section.id`) they reference.

(Further details on agent usage will be added as components are developed.)