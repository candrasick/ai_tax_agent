.PHONY: db-migrate download-bulletins analyze-size parse-tax-code parse-tax-code-custom

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