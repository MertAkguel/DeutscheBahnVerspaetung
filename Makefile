# Short names for the commands used most. "make help" lists them.
# Recipes must be indented with a TAB, not spaces.

# Default month; override with: make train MONTH=2026-11
MONTH ?= 2026-10

.PHONY: help install lint format test check up down logs train

help:  ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install all packages and the git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check code style and likely bugs
	uv run ruff check
	uv run ruff format --check

format:  ## Fix style automatically
	uv run ruff check --fix
	uv run ruff format

test:  ## Run the tests
	uv run pytest

check: lint test  ## Everything CI runs

up:  ## Start Airflow, Postgres and MLflow (Airflow: localhost:8080, MLflow: localhost:5000)
	docker compose up -d --build

down:  ## Stop the stack (data stays)
	docker compose down

logs:  ## Follow the logs of the scheduler
	docker compose logs -f airflow-scheduler

train:  ## Train and register one month by hand, e.g. make train MONTH=2026-11
	run_id=$$(uv run python -m db_delay_mlops.models.train --valid-month $(MONTH) | tail -1) && \
	uv run python -m db_delay_mlops.models.register --run-id $$run_id --valid-month $(MONTH)
