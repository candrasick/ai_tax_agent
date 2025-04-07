.PHONY: db-migrate

db-migrate:
	@echo "Applying database migrations..."
	poetry run alembic upgrade head
	@echo "Database migrations applied." 