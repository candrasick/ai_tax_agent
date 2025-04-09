.PHONY: db-migrate download-bulletins analyze-size parse-tax-code parse-tax-code-custom analyze-amendments analyze-amendments-limit visualize-amendments

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