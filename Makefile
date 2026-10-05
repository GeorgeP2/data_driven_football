PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check new-project notebook clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install core + dev + notebook deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,notebooks]"
	$(BIN)/pre-commit install
	$(BIN)/python -m ipykernel install --user --name football --display-name "Python (football)"

lint: ## Lint with ruff
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-format and fix lint issues
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## Type-check the shared package
	$(BIN)/mypy

test: ## Run all tests (shared package + every project)
	$(BIN)/pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

new-project: ## Scaffold a project: make new-project name="Expected Goals Model" category=ml
	$(BIN)/python scripts/new_project.py "$(name)" --category "$(or $(category),ml)"

notebook: ## Launch JupyterLab
	$(BIN)/jupyter lab

clean: ## Remove caches and build artefacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage htmlcov
