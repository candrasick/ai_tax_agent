.PHONY: db-migrate download-bulletins analyze-size analyze-mentions parse-tax-code parse-tax-code-custom analyze-amendments analyze-amendments-limit visualize-amendments parse-bulletins parse-bulletins-clear link-bulletins link-bulletins-clear test test-unit test-integration test-db-integrity scrape-instructions scrape-instructions-clear extract-form-fields extract-form-fields-clear

db-migrate:
	@echo "Applying database migrations..."
	poetry run alembic upgrade head
	@echo "Database migrations applied."

download-bulletins:
	@echo "Downloading IRS IRB bulletins..."
	poetry run python scripts/download_irb_bulletins.py
	@echo "Finished downloading bulletins."

analyze-size:
	@echo "Analyzing tax code and bulletin page counts..."
	poetry run python scripts/analyze_code_size.py
	@echo "Analysis complete. Plot saved to plots/tax_code_growth.png (by default)."

analyze-mentions:
	@echo "Analyzing section mentions in bulletins and amendments in IRC..."
	poetry run python scripts/analyze_section_mentions.py
	@echo "Mention analysis complete. Plots saved to plots/ (by default)."

parse-tax-code:
	@echo "Parsing U.S. Tax Code XML file..."
	poetry run python scripts/parse_tax_code.py
	@echo "Tax code parsing complete."

parse-tax-code-custom:
	@echo "Parsing custom U.S. Tax Code XML file..."
	@read -p "Enter the path to the XML file: " xml_file; \
	poetry run python scripts/parse_tax_code.py --xml-file $$xml_file
	@echo "Tax code parsing complete."

analyze-amendments:
	@echo "Running LLM analysis to count amendments and clean section text..."
	poetry run python scripts/analyze_amendments.py
	@echo "Amendment analysis complete."

analyze-amendments-limit:
	@echo "Running LLM analysis (limited)..."
	@read -p "Enter the maximum number of sections to process: " limit; \
	poetry run python scripts/analyze_amendments.py --limit $$limit
	@echo "Limited amendment analysis complete."

visualize-amendments:
	@echo "Generating amendment data visualizations..."
	poetry run python scripts/visualize_amendments.py
	@echo "Visualizations saved to plots/ directory."

parse-bulletins:
	@echo "Parsing IRS Bulletin PDFs (incremental)..."
	poetry run python scripts/parse_bulletins.py
	@echo "Bulletin parsing complete."

parse-bulletins-clear:
	@echo "Clearing existing IRS Bulletin data and parsing PDFs..."
	poetry run python scripts/parse_bulletins.py --clear
	@echo "Bulletin parsing complete."

link-bulletins:
	@echo "Linking bulletin items to code sections..."
	poetry run python scripts/link_bulletins_to_sections.py
	@echo "Linking complete."

link-bulletins-clear:
	@echo "Clearing existing links and linking bulletin items to code sections..."
	poetry run python scripts/link_bulletins_to_sections.py --clear
	@echo "Linking complete."

test:
	@echo "Running all tests..."

test-unit:
	@echo "Running unit tests..."
	poetry run pytest -sv tests/unit
	@echo "Unit tests complete."

test-integration:
	@echo "Running integration tests..."
	poetry run pytest -sv tests/integration
	@echo "Integration tests complete."

test-db-integrity:
	@echo "Running database integrity tests..."
	poetry run pytest -sv tests/integration/test_db_integrity.py
	@echo "Database integrity tests complete."

scrape-instructions:
	@echo "Scraping IRS form instructions (incremental update)..."
	poetry run python scripts/scrape_irs_instructions.py
	@echo "Scraping complete. Data saved to database (tax_agent.db)"

scrape-instructions-clear:
	@echo "Clearing existing form instructions and scraping new ones..."
	poetry run python scripts/scrape_irs_instructions.py --clear --force
	@echo "Scraping complete. Data saved to database (tax_agent.db)"

extract-form-fields:
	@echo "Extracting form fields from IRS instructions..."
	poetry run python scripts/extract_form_fields.py
	@echo "Extraction complete. Fields saved to database."

extract-form-fields-clear:
	@echo "Clearing existing form fields and extracting new ones..."
	poetry run python scripts/extract_form_fields.py --clear
	@echo "Extraction complete. Fields saved to database." 