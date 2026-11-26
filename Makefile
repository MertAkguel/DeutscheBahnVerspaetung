# Short names for the commands used most. "make help" lists them.
# Recipes must be indented with a TAB, not spaces.

# On Windows, make would run recipes in cmd.exe, which has no grep, tail, mkdir -p or $(...).
# Use the sh.exe that ships with Git for Windows instead, so the same recipes work everywhere.
# Elsewhere installed? Override with: make GIT_SH=D:/Tools/Git/bin/sh.exe <target>
ifeq ($(OS),Windows_NT)
GIT_SH ?= C:/PROGRA~1/Git/bin/sh.exe
ifeq ($(wildcard $(GIT_SH)),)
$(error Git Bash not found at $(GIT_SH). Install Git for Windows or pass GIT_SH=<path to sh.exe>)
endif
SHELL := $(GIT_SH)
endif

# Default month; override with: make data MONTH=2026-11
MONTH ?= 2026-10

.PHONY: help install lint format test integration check data up down restart-api logs train api demo

help:  ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install all packages and the git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check code style and likely bugs
	uv run ruff check
	uv run ruff format --check

format:  ## Fix style automatically
	uv run ruff check --fix
	uv run ruff format

test:  ## Run the unit tests (seconds, no Docker, no internet)
	uv run pytest -m "not integration"

integration:  ## Test the running stack end to end (needs: make up)
	uv run pytest -m integration -v

check: lint test  ## Everything CI runs

data:  ## Download, build and validate one month, e.g. make data MONTH=2026-11
	uv run python -m db_delay_mlops.data.download --start $(MONTH)
	uv run python -m db_delay_mlops.features.build --month $(MONTH)
	uv run python -m db_delay_mlops.data.validate --month $(MONTH)

up:  ## Start everything: Airflow :8080, MLflow :5000, API :8000, demo :8501
	mkdir -p data/predictions logs mlflow_store  # created by you, not by Docker (which would make them root's)
	docker compose up -d --build

down:  ## Stop the stack (data, runs and models stay)
	docker compose down

restart-api:  ## Reload the API after a new champion
	docker compose restart api

logs:  ## Follow the logs of the scheduler
	docker compose logs -f airflow-scheduler

train:  ## Train and register one month by hand, e.g. make train MONTH=2026-11
	run_id=$$(uv run python -m db_delay_mlops.models.train --valid-month $(MONTH) | tail -1) && \
	uv run python -m db_delay_mlops.models.register --run-id $$run_id --valid-month $(MONTH)

api:  ## Run the API locally with auto-reload, for development (MLflow must run)
	uv run uvicorn db_delay_mlops.api.main:app --reload --port 8000

demo:  ## Run the Streamlit demo locally, for development (the API must run)
	uv run streamlit run src/db_delay_mlops/demo/app.py
