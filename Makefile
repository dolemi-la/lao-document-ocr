PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn

.PHONY: install dev test lint format api web docker-up docker-down capture-suite capture-phone capture-flatbed capture-phone-pack capture-flatbed-pack

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


capture-suite:
	./scripts/generate_public_capture_suite.sh


CAPTURE_KIT ?= benchmarks/capture-packs/project-authored-lao-v1.collector.zip
CAPTURE_ROOT ?= benchmarks/private/captures
SUBMISSION_ROOT ?= benchmarks/private/submissions
CAPTURE_HOST ?= 0.0.0.0
PHONE_CAPTURE_ID ?= phone-a
PHONE_CAPTURE_PORT ?= 8090
FLATBED_CAPTURE_ID ?= flatbed-a
FLATBED_CAPTURE_PORT ?= 8091

capture-phone:
	$(VENV)/bin/lao-ocr serve-capture-kit \
		--kit $(CAPTURE_KIT) \
		--output-dir $(CAPTURE_ROOT)/$(PHONE_CAPTURE_ID) \
		--capture-id $(PHONE_CAPTURE_ID) \
		--mode phone-photo \
		--host $(CAPTURE_HOST) \
		--port $(PHONE_CAPTURE_PORT)

capture-flatbed:
	$(VENV)/bin/lao-ocr serve-capture-kit \
		--kit $(CAPTURE_KIT) \
		--output-dir $(CAPTURE_ROOT)/$(FLATBED_CAPTURE_ID) \
		--capture-id $(FLATBED_CAPTURE_ID) \
		--mode flatbed-scan \
		--host $(CAPTURE_HOST) \
		--port $(FLATBED_CAPTURE_PORT)

capture-phone-pack:
	mkdir -p $(SUBMISSION_ROOT)
	$(VENV)/bin/lao-ocr pack-capture-submission \
		--capture-dir $(CAPTURE_ROOT)/$(PHONE_CAPTURE_ID) \
		--output $(SUBMISSION_ROOT)/$(PHONE_CAPTURE_ID).zip \
		--require-complete

capture-flatbed-pack:
	mkdir -p $(SUBMISSION_ROOT)
	$(VENV)/bin/lao-ocr pack-capture-submission \
		--capture-dir $(CAPTURE_ROOT)/$(FLATBED_CAPTURE_ID) \
		--output $(SUBMISSION_ROOT)/$(FLATBED_CAPTURE_ID).zip \
		--require-complete
