# Short aliases for the commands documented in docs/development.md.
# Recipes are plain `uv` / `docker compose` calls, so they work under sh (Linux, CI, Colab) and cmd (Windows).

COMPOSE = docker compose -f deploy/compose/docker-compose.yml

.PHONY: help setup test lint format models models-all up down ps logs

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

up:  ## Start the local stack and wait until healthy
	$(COMPOSE) up -d --wait

down:  ## Stop the local stack
	$(COMPOSE) down

ps:  ## Show stack status
	$(COMPOSE) ps

logs:  ## Follow stack logs
	$(COMPOSE) logs -f --tail=100
