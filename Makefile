PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn

.PHONY: install dev test lint format api web docker-up docker-down

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev]"

dev:
	docker compose up --build

test:
	$(PYTEST)

lint:
	$(RUFF) check src services tests
	cd apps/web && pnpm lint

format:
	$(RUFF) format src services tests
	$(RUFF) check --fix src services tests

api:
	PYTHONPATH=src $(UVICORN) services.api.app.main:app --reload --port 8000

web:
	cd apps/web && pnpm dev

docker-up:
	docker compose up --build

docker-down:
	docker compose down
