# Common dev commands. Run `make help` to see them.
# Targets are run via uv so the .venv stays consistent for everyone.

.DEFAULT_GOAL := help
.PHONY: help install lint lint-fix fmt fmt-check type test cov pre-commit check clean

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install dev dependencies and the git pre-commit hook.
	uv sync --group dev
	uv run pre-commit install

lint:  ## Run ruff lint (no fixes).
	uv run ruff check .

lint-fix:  ## Run ruff lint with auto-fix.
	uv run ruff check . --fix

fmt:  ## Format the codebase with ruff.
	uv run ruff format .

fmt-check:  ## Check formatting without modifying files.
	uv run ruff format --check .

type:  ## Run ty type checker (advisory).
	uv run ty check .

test:  ## Run the test suite.
	uv run pytest

cov:  ## Run tests with coverage report.
	uv run pytest --cov

pre-commit:  ## Run all pre-commit hooks against every file.
	uv run pre-commit run --all-files

check: lint fmt-check test  ## Run lint + format check + tests (no auto-fix).

clean:  ## Remove caches and coverage artifacts.
	rm -rf .ruff_cache .ty_cache .pytest_cache .coverage .coverage.* htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
