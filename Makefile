# Short aliases for the commands documented in docs/development.md.
# Recipes are plain `uv` / `docker compose` calls, so they work under sh (Linux, CI, Colab) and cmd (Windows).

.PHONY: help setup test lint format models models-all

help:  ## List targets
	@uv run python -c "import re; [print(f\"{m[0]:<12} {m[1]}\") for m in re.findall(r\"^([a-z-]+):.*## (.*)$$\", open(\"Makefile\").read(), re.M)]"

setup:  ## Create .venv and install exact versions from uv.lock
	uv sync

test:  ## Run unit tests (no model or Docker needed)
	uv run pytest -m "not contract and not integration"

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

models:  ## Download the pinned CI + dev GGUF models to ~/models
	uv run python scripts/download_models.py

models-all:  ## Also download benchmark-only models
	uv run python scripts/download_models.py --all
