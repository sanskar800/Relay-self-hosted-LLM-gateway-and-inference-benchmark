# Short aliases for the commands documented in docs/development.md.
# Recipes are plain `uv` / `docker compose` calls, so they work under sh (Linux, CI, Colab) and cmd (Windows).

COMPOSE = docker compose -f deploy/compose/docker-compose.yml

.PHONY: help setup test lint format models models-all up up-gpu up-all down ps logs run serve test-contract test-integration test-alerts db-shell redis-cli db-reset db-apply bench bench-overhead bench-stack-gpu bench-stack-cpu report

help:  ## List targets
	@uv run python -c "import re; [print(f\"{m[0]:<12} {m[1]}\") for m in re.findall(r\"^([a-z-]+):.*## (.*)$$\", open(\"Makefile\").read(), re.M)]"

setup:  ## Create .venv and install exact versions from uv.lock
	uv sync

test:  ## Run unit tests (no model or Docker needed)
	uv run pytest -m "not contract and not integration"

test-contract:  ## OpenAI SDK contract tests (needs `make up` and `make run`)
	uv run pytest -m contract -v

test-integration:  ## Tests against the Compose services (needs `make up`)
	uv run pytest -m integration -v

# Benchmarks: make bench SCENARIO=bench/scenarios/<name>.yaml [ARGS=--scratch]
SCENARIO ?= bench/scenarios/local-smoke.yaml
ARGS ?=

bench:  ## Run a benchmark scenario (SCENARIO=..., ARGS=--scratch for a smoke run)
	uv run python -m bench.loadgen $(SCENARIO) $(ARGS)

bench-overhead:  ## Measure Relay's added latency against a fixed-latency fake engine (needs `make up`)
	uv run python -m bench.overhead $(ARGS)

report:  ## Regenerate docs/report.md, results/charts and per-run summaries from results/
	uv run python -m bench.analysis.analyze

# Lean stack for timed runs (docs/findings.md F8): only the engine under test plus what
# Relay needs; the other engine and the observability services are stopped. Engine
# settings are fixed for every benchmark (ADR-001). Pick the GPU model with
# LLAMACPP_GPU_MODEL / LLAMACPP_GPU_ALIAS.
BENCH_SLOTS = LLAMACPP_PARALLEL=4 LLAMACPP_CTX=4096 LLAMACPP_GPU_PARALLEL=4 LLAMACPP_GPU_CTX=4096 LLAMACPP_CACHE_RAM=0

bench-stack-gpu:  ## Lean stack for GPU runs: llama.cpp GPU + Postgres + Redis only
	$(COMPOSE) stop llamacpp prometheus grafana jaeger
	$(BENCH_SLOTS) $(COMPOSE) --profile gpu up -d --wait --force-recreate llamacpp-gpu postgres redis

bench-stack-cpu:  ## Lean stack for the CPU baseline: llama.cpp CPU + Postgres + Redis only
	$(COMPOSE) --profile gpu stop llamacpp-gpu prometheus grafana jaeger
	$(BENCH_SLOTS) $(COMPOSE) up -d --wait --force-recreate llamacpp postgres redis

test-alerts:  ## Validate Prometheus config and unit-test the alert rules (promtool in Docker)
	docker run --rm --entrypoint promtool -v "$(CURDIR)/monitoring:/etc/prometheus:ro" -w //etc/prometheus prom/prometheus:v3.15.0 check config prometheus.yml
	docker run --rm --entrypoint promtool -v "$(CURDIR)/monitoring:/etc/prometheus:ro" -w //etc/prometheus prom/prometheus:v3.15.0 test rules rules.test.yml

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

up-gpu:  ## Also start llama.cpp on the RTX 4050 (:8082)
	$(COMPOSE) --profile gpu up -d --wait

up-all:  ## Everything in Docker, including Relay itself (built from the Dockerfile)
	$(COMPOSE) --profile app up -d --build --wait

down:  ## Stop the local stack (including the GPU and Relay containers)
	$(COMPOSE) --profile gpu --profile app down

ps:  ## Show stack status
	$(COMPOSE) ps

logs:  ## Follow stack logs
	$(COMPOSE) logs -f --tail=100

serve:  ## Start Relay on :8000 production-style (graceful shutdown, no auto-reload)
	uv run python -m relay.serve

run:  ## Start Relay on http://localhost:8000 (auto-reload on code changes)
	uv run uvicorn relay.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir src

db-shell:  ## psql into Relay's Postgres
	$(COMPOSE) exec postgres psql -U relay -d relay

redis-cli:  ## redis-cli into Relay's Redis
	$(COMPOSE) exec redis redis-cli

db-reset:  ## DELETE Relay's Postgres data and re-create it from initdb/*.sql
	$(COMPOSE) rm --stop --force postgres
	docker volume rm --force relay_pgdata
	$(COMPOSE) up -d --wait postgres

db-apply:  ## Apply idempotent SQL (views) to the existing database without resetting it
	$(COMPOSE) exec -T postgres psql -U relay -d relay -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/002_usage_daily.sql
