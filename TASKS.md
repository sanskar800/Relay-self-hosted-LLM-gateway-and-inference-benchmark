# Relay: one-week execution plan

**Budget:** about 30–35 focused hours over 7 days (about 5 h/day). **Spend:** £0 (no GPU rental, no paid APIs).
**Priority tags:** `[MUST]` is required for the week to count as done. `[SHOULD]` is production quality, if time allows. `[OPTIONAL]` is only attempted once every MUST is green.

**Rules:** build first, benchmark last. No invented numbers; anything unmeasured is `TBD`. If a day overruns, cut SHOULD/OPTIONAL items, never MUST ones, and adjust later days here.

Legend: each task lists **Why**, **Expected result**, **Files** and **Verify**.

---

## Decisions needed before / on Day 1

- [x] `[MUST]` **D0.1 Approve spec deviations** (approved 2026-10-08: local RTX 4050 + free Colab, no paid GPU) (see [docs/architecture.md §6](docs/architecture.md#6-deviations-from-the-original-specification))
    - Why: no paid GPU, 6 GB VRAM, one week. Changing the spec silently would make the README dishonest.
    - Expected result: deviations accepted; the table in architecture.md is marked "Accepted".
    - Files: `docs/architecture.md`
    - Verify: the table status is updated.
- [x] `[MUST]` **D0.2 Decide repo location**: moved to `C:\dev\5projects\relay` on 2026-10-08
    - Why: OneDrive syncing `.venv`/models breaks git/uv and slows Docker mounts.
    - Expected result: project at its final path, with OneDrive no longer syncing it.
    - Files: none (folder move)
    - Verify: `git status` works and the OneDrive tray shows no sync activity for the folder.

---

## Day 1: Foundations (≈5 h)

Goal: **a request from the OpenAI SDK travels Client → Relay → llama.cpp → back.**

- [x] `[MUST]` **1.1 Write the one-page requirements brief** (done 2026-10-08)
    - Why: an SLO gives the benchmark a pass/fail target ("TTFT p95 < 1 s at N concurrent users"); tenants and quotas shape the data model.
    - Expected result: `docs/brief.md` with users, tenants, quotas, the target SLO (labelled *target*, not a result) and what is out of scope.
    - Files: `docs/brief.md`
    - Verify: it fits on one page; every later feature traces back to a line in it.
- [x] `[MUST]` **1.2 Initialise git and GitHub** (done 2026-10-08, pushed to `main`)
    - Why: version history; CI later; the portfolio lives on GitHub.
    - Expected result: `git init`, `.gitignore` (venv, models, `.env`, caches), `.gitattributes` (LF line endings), first commit, remote set to https://github.com/sanskar800/Relay-self-hosted-LLM-gateway-and-inference-benchmark (exists, empty).
    - Files: `.gitignore`, `.gitattributes`
    - Verify: `git log` shows the commit; `git push` succeeds.
- [x] `[MUST]` **1.3 Verify make, Docker Desktop + WSL2 and NVIDIA GPU support** (done 2026-10-08) (external setup; walk through together)
    - Why: `make` targets are the house style; GPU-in-Docker decides whether llama.cpp uses the RTX 4050 (1.7, Day 6).
    - Expected result: `make --version` works in a fresh terminal (winget's `bin` folder is on the user PATH; VS Code must be fully restarted to pick it up); Docker uses the WSL2 backend; `docker run --rm --gpus all nvidia/cuda:… nvidia-smi` prints the RTX 4050.
    - Files: `docs/development.md` (results recorded)
    - Verify: both commands' output.
- [x] `[MUST]` **1.4 Set up the Python project with uv** (done 2026-10-08)
    - Why: reproducible environment (lock file), src layout, linting and tests from the start.
    - Expected result: `pyproject.toml` (FastAPI, httpx, pydantic-settings, uvicorn; dev: pytest, pytest-asyncio, ruff, openai), `.python-version` = 3.12, `uv.lock`, `src/relay/__init__.py`, `tests/`.
    - Files: `pyproject.toml`, `uv.lock`, `.python-version`, `src/relay/`, `tests/`
    - Verify: `uv sync && uv run pytest && uv run ruff check .` all pass.
- [x] `[MUST]` **1.5 Write a model download script** (done 2026-10-08)
    - Why: model choice must be reproducible from the repo, not "I downloaded something once".
    - Expected result: `scripts/download_models.py` fetches pinned GGUF files (0.5B for CI, 1.5B Q4_K_M for dev) to `~/models` via `huggingface_hub`; `make models`.
    - Files: `scripts/download_models.py`, `Makefile`
    - Verify: files exist with the expected size; licence (Apache-2.0) is noted in development.md.
- [x] `[MUST]` **1.6 Run llama.cpp server in Docker Compose (CPU first)** (done 2026-10-08)
    - Why: Relay needs a backend. Learn the OpenAI chat-completions request/response format and SSE stream format directly from the engine.
    - Expected result: `deploy/compose/docker-compose.yml` with a `llamacpp` service on `:8081`, models mounted read-only.
    - Files: `deploy/compose/docker-compose.yml`, `Makefile`
    - Verify: `curl :8081/v1/chat/completions` returns a completion; with `"stream": true` we see `data: {...}` lines ending in `data: [DONE]`.
- [x] `[SHOULD]` **1.7 llama.cpp on the RTX 4050 (CUDA image)** (done 2026-10-08, after updating the NVIDIA driver to 617.42)
    - Why: the realistic local GPU path; needed for Day 6.
    - Expected result: a `llamacpp-gpu` Compose profile with GPU device reservation and `--n-gpu-layers` set.
    - Files: `deploy/compose/docker-compose.yml`
    - Verify: VRAM use rises when the container starts; `--list-devices` shows the GPU; utilisation rises during generation. (Only an observation, **not** a benchmark number.)
- [x] `[MUST]` **1.8 Minimal Relay: FastAPI app with non-streaming proxy** (done 2026-10-08)
    - Why: the smallest possible gateway, so every later feature has something to attach to.
    - Expected result: `GET /healthz`; `POST /v1/chat/completions` forwards the JSON body to llama.cpp with a shared `httpx.AsyncClient` (created in the app lifespan) and returns the response.
    - Files: `src/relay/main.py`, `src/relay/config.py`, `.env.example`
    - Verify: `curl :8000/v1/chat/completions` returns the model's answer.
- [x] `[MUST]` **1.9 End-to-end with the official OpenAI SDK** (done 2026-10-08)
    - Why: compatibility should be proven, not assumed. This becomes the first contract test.
    - Expected result: `tests/contract/test_openai_sdk.py` uses `OpenAI(base_url="http://localhost:8000/v1")`.
    - Files: `tests/contract/`
    - Verify: `uv run pytest tests/contract` passes with the stack running.
- [x] `[MUST]` **1.10 Update docs and commit** (done 2026-10-08; verified from a clean clone)
    - Expected result: quick start in README and development.md actually works from a clean checkout.
    - Verify: follow the README in a fresh terminal.

**End of Day 1 checkpoint:** OpenAI SDK → Relay → llama.cpp works (non-streaming).

---

## Day 2: Gateway (≈5 h)

Goal: **a correct streaming, OpenAI-compatible gateway with a backend abstraction.**

- [x] `[MUST]` **2.1 OpenAI request/response schemas (Pydantic)** (done 2026-10-10)
    - Why: validate input at the edge; reject bad requests before they reach a GPU.
    - Expected result: models for `ChatCompletionRequest` (the subset we support, with unknown fields passed through), responses and error objects.
    - Files: `src/relay/api/schemas.py`
    - Verify: unit tests for valid, invalid and extra-field payloads.
- [x] `[MUST]` **2.2 SSE streaming passthrough** (done 2026-10-10)
    - Why: streaming is what makes LLM UX tolerable, and it is the hardest part of a proxy to get right (flushing, backpressure, disconnects).
    - Expected result: `stream: true` uses `httpx` `client.stream()` and FastAPI `StreamingResponse`; chunks are forwarded as they arrive; client disconnect cancels the upstream request.
    - Files: `src/relay/api/chat.py`
    - Verify: `curl -N` shows tokens arriving incrementally; the contract test with `stream=True` passes; disconnect test: llama.cpp logs show the request cancelled.
- [x] `[MUST]` **2.3 Backend abstraction** (done 2026-10-10)
    - Why: routing, retries and accounting must not depend on which engine is behind them.
    - Expected result: `Backend` protocol (`chat`, `stream`, `health`, `name`); `OpenAICompatBackend` base; `LlamaCppBackend`, `VLLMBackend` subclasses.
    - Files: `src/relay/backends/{base,openai_compat,llamacpp,vllm}.py`
    - Verify: unit tests with a fake backend.
- [x] `[MUST]` **2.4 Model/backends config (YAML)** (done 2026-10-10)
    - Why: the public model name (`qwen2.5-1.5b-instruct`) maps to an ordered list of backends. This is the basis for routing and fallback.
    - Expected result: `config/relay.yaml` loaded and validated at startup.
    - Files: `config/relay.yaml`, `src/relay/config.py`
    - Verify: startup fails with a clear error on invalid config (tested).
- [x] `[MUST]` **2.5 `GET /v1/models`** (done 2026-10-10)
    - Why: SDKs and tools call it; part of API compatibility.
    - Files: `src/relay/api/models.py`
    - Verify: `client.models.list()` in the contract tests.
- [x] `[MUST]` **2.6 Error handling in OpenAI format** (done 2026-10-10)
    - Why: clients parse `{"error": {...}}`; an upstream 500 should not leak as an HTML page.
    - Expected result: exception handlers; upstream errors mapped (502/503/504); a mid-stream failure emits an SSE error event and closes cleanly.
    - Files: `src/relay/api/errors.py`
    - Verify: unit tests; the SDK raises the right `openai.APIError` subclass.
- [x] `[MUST]` **2.7 Token usage capture** (done 2026-10-10)
    - Why: accounting (Day 3) needs prompt and completion tokens for both streaming and non-streaming requests.
    - Expected result: usage read from the response; for streams, request `stream_options.include_usage` from the backend, with a documented fallback if the engine does not support it.
    - Files: `src/relay/accounting/usage.py`
    - Verify: tests assert the token counts match the engine's report.
- [x] `[MUST]` **2.8 Unit tests with a fake backend** (no model needed) (done 2026-10-10; 103 unit tests in ~2 s)
    - Why: fast, deterministic tests that do not need a running model.
    - Expected result: a fake OpenAI-compatible server (respx or an in-process ASGI app) that streams scripted chunks.
    - Files: `tests/unit/`, `tests/conftest.py`
    - Verify: `uv run pytest tests/unit` runs in under 5 s.
- [x] `[MUST]` **2.9 Request size and parameter limits** (body bytes, `max_tokens` ceiling, message count) (done 2026-10-10; retagged SHOULD → MUST because the spec's security list and the brief's FR3 require request size limits)
    - Files: `src/relay/api/limits.py`
    - Verify: oversized request gets 413/400.
- [x] `[MUST]` **2.10 Colab feasibility check (≈20 min, de-risking only, no benchmarking)** (done 2026-10-10: T4, vLLM 0.31.0 works with `--dtype half`; see development.md §9)
    - Why: the benchmark depends on a free Colab GPU running vLLM. Finding out on Day 6 that it doesn't work would leave no time to fix it.
    - Expected result: in a scratch Colab notebook, the runtime shows a GPU (`nvidia-smi`); `pip install vllm` succeeds; `vllm serve Qwen/Qwen2.5-1.5B-Instruct` answers one curl request. Record the GPU model, vLLM version and any workarounds in development.md.
    - Files: `docs/development.md`
    - Verify: one successful completion from vLLM on Colab. If it fails, revise the Day 6 plan now.
- [ ] `[OPTIONAL]` **2.11 `/v1/completions` and `/v1/embeddings` passthrough**

**Checkpoint:** streaming works through Relay via the OpenAI SDK; unit tests green.

---

## Day 3: Multi-tenancy and governance (≈5 h)

Goal: **every request is attributed to a tenant, limited and recorded.**

- [x] `[MUST]` **3.1 PostgreSQL + Redis in Compose; schema** (done 2026-10-11; Postgres on host port 5433)
    - Why: Postgres is the durable ledger and key store; Redis is the shared rate-limit state.
    - Expected result: services with health checks; `deploy/compose/initdb/001_schema.sql` (tables `tenants`, `api_keys(key_hash, tenant_id, …)`, `usage_events`). Plain SQL, no migration framework (deliberate simplicity).
    - Files: `deploy/compose/`, `src/relay/accounting/db.py`
    - Verify: `psql` shows the tables; `redis-cli ping` returns PONG.
- [x] `[MUST]` **3.2 API keys and tenant identification** (done 2026-10-11)
    - Why: no identity means no quotas and no cost attribution.
    - Expected result: `Authorization: Bearer rk_…`; SHA-256 hash lookup with a short in-memory TTL cache; 401 in OpenAI error format; admin CLI `relay-admin create-tenant / create-key` (shows the key once).
    - Files: `src/relay/auth/`, `src/relay/cli.py`
    - Verify: tests for valid, invalid and revoked keys; the raw key never appears in the DB or logs.
- [x] `[MUST]` **3.3 Per-tenant model policy** (done 2026-10-11)
    - Why: governance. Team A may use model X but not model Y, and has a per-request `max_tokens` cap.
    - Expected result: allow-list and caps checked before routing. ~~403 if not allowed~~ **Changed to 404 `model_not_found`, identical to an unknown model** (as OpenAI does), so tenants cannot discover which models exist; the reason is logged server-side. An omitted `max_tokens` gets the tenant's cap added, so the cap cannot be bypassed.
    - Files: `src/relay/auth/policy.py`
    - Verify: unit tests.
- [x] `[MUST]` **3.4 Redis token-bucket rate limiter (Lua)** (done 2026-10-11)
    - Why: fair sharing of a scarce GPU across tenants and replicas; the Lua script makes "refill + check + take" atomic.
    - Expected result: request/min and token/min buckets per tenant; 429 with `Retry-After` and `x-ratelimit-*` headers.
    - Files: `src/relay/limits/token_bucket.py`, `src/relay/limits/token_bucket.lua`
    - Verify: unit tests of the refill maths (fake clock); integration test against real Redis: N+1 rapid requests and the last one is 429.
- [x] `[MUST]` **3.5 ADR-002: limiter behaviour when Redis is down, and its implementation** (done 2026-10-11: fail open to local per-instance limits)
    - Why: a classic interview question. Fail-open keeps an internal tool available; fail-closed protects a paid API.
    - Expected result: ADR written; the chosen behaviour implemented (likely fail-open to a local in-memory bucket, plus a metric and a warning log).
    - Files: `docs/adr/002-rate-limit-failure-behaviour.md`, `src/relay/limits/`
    - Verify: integration test that stops Redis, sends requests and asserts the documented behaviour.
- [x] `[MUST]` **3.6 Non-blocking usage ledger** (done 2026-10-11; `cost_gbp` NULL until benchmark-based rates exist)
    - Why: cost attribution must be durable, but a slow database must never slow a user's request.
    - Expected result: one usage event per request (tenant, key ID, model, backend, prompt/completion tokens, TTFT, duration, status, estimated £); bounded `asyncio.Queue` and a batch writer task; buffering when Postgres is down; dropped events counted.
    - Files: `src/relay/accounting/ledger.py`, `config/pricing.yaml`
    - Verify: after N requests, `SELECT count(*)` equals N; Postgres-down test shows requests still succeed.
- [x] `[SHOULD]` **3.7 Usage report query** (done 2026-10-11)
    - Expected result: SQL view `usage_daily` (tenant, day, tokens, £) and `relay-admin usage` CLI.
    - Verify: matches what the test traffic generated.
- [ ] `[OPTIONAL]` **3.8 Exact-match response cache** (temperature = 0 only)

**Checkpoint:** two tenants with different limits; one gets 429 while the other is unaffected; the ledger shows both.

---

## Day 4: Reliability and observability (≈5–6 h, the heaviest day)

Goal: **Relay survives backend failure and you can see what it is doing.**

- [x] `[MUST]` **4.1 Per-backend timeouts** (done 2026-10-13) (connect, first byte, idle between chunks, total)
    - Why: a hung backend must not hold client connections forever.
    - Files: `src/relay/backends/base.py`, `config/relay.yaml`
    - Verify: a fake slow backend triggers 504 within the configured time.
- [x] `[MUST]` **4.2 Retries only before the first token (ADR-003)** (done 2026-10-13)
    - Why: once tokens reach the user, a retry would produce duplicated or contradictory text.
    - Expected result: retry with exponential backoff + jitter on connect errors / 503 only, before any byte is sent; mid-stream failure leads to an SSE error event and no retry.
    - Files: `src/relay/routing/retry.py`, `docs/adr/003-streaming-retry-policy.md`
    - Verify: tests "fails before first token, so retried" and "fails after 3 tokens, so clean error, no retry".
- [x] `[MUST]` **4.3 Circuit breaker** (done 2026-10-13)
    - Why: stop sending traffic to a backend that is clearly down; probe it to recover.
    - Expected result: closed → open → half-open state machine per backend, with configurable thresholds.
    - Files: `src/relay/routing/circuit_breaker.py`
    - Verify: unit tests drive every state transition with a fake clock.
- [x] `[MUST]` **4.4 Routing and fallback** (done 2026-10-13)
    - Why: availability. If the primary is down, use the next backend for that model.
    - Expected result: the router picks the first backend whose breaker is not open; falls back on pre-first-token failure. Local demo: llama.cpp GPU primary, llama.cpp CPU secondary.
    - Files: `src/relay/routing/router.py`
    - Verify: `docker stop` the primary; requests still succeed via the secondary; the breaker metric shows open.
- [x] `[MUST]` **4.5 Graceful shutdown** (done 2026-10-13; `make serve`)
    - Why: deploys should not cut users off mid-answer.
    - Expected result: on SIGTERM, `/readyz` returns 503, new requests are rejected and in-flight streams drain up to a deadline.
    - Files: `src/relay/main.py`
    - Verify: start a long stream, send SIGTERM, and the stream completes.
- [x] `[MUST]` **4.6 Structured JSON logs** (done 2026-10-13)
    - Expected result: structlog JSON with request ID, tenant, model, backend, status and durations; **no prompt bodies** by default.
    - Files: `src/relay/telemetry/logging.py`
    - Verify: log line inspection; a test asserts the prompt text is absent.
- [x] `[MUST]` **4.7 Prometheus metrics** (done 2026-10-13)
    - Why: latency percentiles and per-tenant usage are the product.
    - Expected result: `/metrics` with request count by status/backend, TTFT and ITL histograms, output tokens, gateway overhead histogram, in-flight per backend, breaker state gauge, tokens and £ per tenant, ledger drops.
    - Files: `src/relay/telemetry/metrics.py`
    - Verify: `curl :8000/metrics` after traffic.
- [x] `[MUST]` **4.8 Prometheus + Grafana in Compose with a provisioned dashboard** (done 2026-10-13)
    - Expected result: dashboard JSON committed: traffic, errors, TTFT/ITL p50/p95/p99, tokens/s, per-tenant usage, breaker state.
    - Files: `monitoring/prometheus.yml`, `monitoring/grafana/`
    - Verify: `make up`, generate traffic, panels move.
- [x] `[MUST]` **4.9 Basic OpenTelemetry tracing → Jaeger** (done 2026-10-13)
    - Why: shows where time goes inside a single request.
    - Expected result: FastAPI + httpx auto-instrumentation plus manual spans for auth, limit and route; exported via OTLP to Jaeger in Compose.
    - Files: `src/relay/telemetry/tracing.py`, compose
    - Verify: a trace in the Jaeger UI shows the stage spans.
- [x] `[MUST]` **4.10 Failure-scenario integration tests** (done 2026-10-13)
    - Expected result: backend killed mid-stream; slow backend → timeout → fallback; Redis down → documented behaviour; Postgres down → requests still served.
    - Files: `tests/integration/`
    - Verify: `make test-integration` is green.
- [x] `[SHOULD]` **4.11 Relay Dockerfile; `make up` runs the whole stack** (done 2026-10-13)
    - Files: `Dockerfile`, compose
    - Verify: fresh clone, then `make up`, then curl works.
- [x] `[SHOULD]` **4.12 Prometheus alert rules** (error rate, TTFT p95 above SLO) (done 2026-10-13)
    - Files: `monitoring/rules.yml`

**Checkpoint:** kill a backend under traffic; the dashboard shows the breaker opening and fallback taking over.

---

## Day 5: Benchmark harness (≈5 h)

Goal: **a trustworthy measuring instrument, validated before it measures anything real.**

- [x] `[MUST]` **5.1 Scenario config schema (YAML + Pydantic)** (done 2026-10-14)
    - Expected result: target URL, model, arrival rates (req/s), duration, warm-up, prompt/output length distribution, seed, SLO.
    - Files: `bench/scenarios/*.yaml`, `bench/loadgen/config.py`
    - Verify: invalid configs are rejected.
- [x] `[MUST]` **5.2 Prompt/length mix** (done 2026-10-14)
    - Why: benchmarks with one fixed prompt length flatter the engine.
    - Expected result: seeded synthetic mix with a documented length distribution. `[SHOULD]`: lengths sampled from a public conversation dataset.
    - Files: `bench/loadgen/workload.py`
    - Verify: a histogram of generated lengths matches the documented distribution.
- [x] `[MUST]` **5.3 Open-loop async load generator (Poisson arrivals)** (done 2026-10-14)
    - Why: closed-loop testing hides overload because the client slows down with the server.
    - Expected result: arrivals scheduled independently of completions; per-request timestamps (send, first token, every token, end), token counts, status.
    - Files: `bench/loadgen/runner.py`
    - Verify: at a fixed rate λ, the observed mean inter-arrival time ≈ 1/λ.
- [x] `[MUST]` **5.4 Metric computation** (done 2026-10-14)
    - Expected result: TTFT, ITL, TPOT, E2E latency, request throughput, output tokens/s, goodput (share meeting the SLO), error rate; p50/p95/p99.
    - Files: `bench/loadgen/metrics.py`
    - Verify: unit tests on hand-built timelines with known answers.
- [x] `[MUST]` **5.5 Harness self-validation against a fake backend with known latency** (ground truth = the engine's recorded timeline, not its scripted delays: Windows sleeps overshoot)
    - Why: proves the instrument is accurate before trusting it on a real model.
    - Expected result: a fake server with injected TTFT/ITL; the harness recovers those values within tolerance.
    - Files: `tests/integration/test_harness_accuracy.py`
    - Verify: test passes.
- [x] `[MUST]` **5.6 Gateway overhead measurement** (`python -m bench.overhead`: fixed-latency fake engine, per-request time outside the engine, cross-checked with `relay_gateway_overhead_seconds`; real figures come from the Day 6 run)
    - Expected result: the same scenario run direct-to-backend and via Relay; plus a server-side overhead histogram.
    - Verify: both paths produce results; the difference is reported (value TBD until measured).
- [x] `[MUST]` **5.7 Results storage** (folder name also carries `-HHMMSS`, so two runs on one day never collide; also `summary.json`)
    - Expected result: `results/run-YYYY-MM-DD-<name>/` containing `config.yaml`, `results.parquet` (one row per request), `env.json` (GPU, driver, engine version, git SHA).
    - Files: `bench/loadgen/results.py`
    - Verify: `duckdb -c "select count(*) from 'results/*/results.parquet'"`.
- [x] `[MUST]` **5.8 Cost model**
    - Expected result: `£/1M output tokens` from measured tokens/s and `bench/pricing.yaml` (GPU class, £/h, source URL, date). Prices are labelled as assumptions.
    - Files: `bench/analysis/cost.py`, `bench/pricing.yaml`
    - Verify: unit test of the formula.
- [x] `[MUST]` **5.9 Local dry run** (1.5B model, llama.cpp, low rates) (done 2026-10-16: `make bench` open and closed, CPU, results in `_scratch`; found and fixed cross-run prompt caching)
    - Verify: a complete results folder is produced end to end. Treat this as a smoke test, not headline numbers.
- [x] `[SHOULD]` **5.10 GSM8K quality evaluation** (done with 6.5, on the Colab configurations) (fixed 200-question subset, exact match, temperature 0, via Relay)
    - Files: `bench/quality/gsm8k.py`
    - Verify: runs on the dev model; accuracy recorded only from the actual run.
- [x] `[SHOULD]` **5.11 Closed-loop mode** (to demonstrate the overload-hiding effect in the report) (`mode: closed`, `concurrency: [...]`; `bench/scenarios/local-smoke-closed.yaml`)

**Checkpoint:** `make bench SCENARIO=bench/scenarios/local-smoke.yaml` produces a valid results folder.

---

## Day 6: GPU benchmark (≈5 h, £0)

Goal: **real measurements on hardware we actually have.** We only compare configurations that actually ran.

- [x] `[MUST]` **6.1 Fix the final configuration matrix and write ADR-001** (done 2026-10-16)
    - Why: decide what is feasible on a 6 GB RTX 4050 and the free Colab GPU before spending time.
    - Candidate matrix (to confirm):
        - Local RTX 4050: llama.cpp CUDA, Qwen2.5-1.5B Q4_K_M vs Q8_0; 7B Q4_K_M if it fits. Compare with the llama.cpp CPU baseline.
        - Colab GPU: vLLM, Qwen2.5-1.5B FP16 vs AWQ; 7B AWQ; `max_num_seqs` sweep; prefix caching on/off `[SHOULD]`.
    - Files: `docs/adr/001-inference-engines-and-hardware.md`, `bench/scenarios/`
    - Verify: each config has a scenario file.
- [~] `[MUST]` **6.2 Local GPU benchmark runs** (**dropped 2026-10-25**, accepted deviation: architecture.md §6, ADR-001 amendment; tooling kept, unrun)
    - Expected result: results folders for each local config across several arrival rates.
    - Verify: the Parquet row count matches requests sent; error rate recorded.
- [x] `[MUST]` **6.3 Colab notebook** (built 2026-10-22; dry-run in Linux with a stub vLLM; first real Colab run 2026-10-22)
    - Known from the 2.10 check: start vLLM with `--dtype half --port 8001` (T4 has no bfloat16; 8000 is Relay's port); Colab runs Python 3.13, uv installs the project's 3.12.
    - Expected result: `bench/colab/relay_benchmark.ipynb` does: clone repo → `uv`/pip install → detect GPU → start vLLM (and Relay in minimal mode: in-memory limiter, ledger off) → run scenarios → zip `results/` → download or save to Drive. No code edits inside Colab.
    - Files: `bench/colab/`, `scripts/colab_run.sh`
    - Verify: runs top to bottom on a fresh runtime.
- [x] `[MUST]` **6.4 Colab vLLM runs** (done 2026-10-24: 1.5B FP16, 1.5B AWQ, 7B AWQ on a T4; findings F10-F11)
    - Expected result: results for at least 3 engine/quantisation configs (MVP requirement), with GPU model recorded in `env.json`.
    - Verify: results are committed under `results/`.
- [x] `[SHOULD]` **6.5 GSM8K per quantisation config** (done 2026-10-25 on Colab: findings F12; `bench/quality/`; covers 5.10)
- [ ] `[SHOULD]` **6.6 Grafana screenshot under load** (local stack) (moved to Day 7, with the README)
- [~] `[OPTIONAL]` **6.7 vLLM locally on the RTX 4050 via WSL/Docker** (skipped: local benchmarking dropped, ADR-001 amendment) (small model; FP8 is possible on Ada)

**Known risks:** free Colab GPUs are not guaranteed (retry later or use a different time of day); sessions can disconnect, so save results after each scenario; recent vLLM releases may drop support for older GPUs (check compute capability 7.5 support; pin a version if needed).

---

## Day 7: Analysis and portfolio polish (≈5 h)

- [x] `[MUST]` **7.1 DuckDB analysis over all runs** (SQL recomputes every metric from raw timestamps; matches the harness to 1e-9, tested)
    - Files: `bench/analysis/queries.sql`, `bench/analysis/analyze.py`
    - Verify: the script regenerates every table from Parquet.
- [x] `[MUST]` **7.2 Charts** (`results/charts/`, palette validated) (TTFT p95 vs arrival rate; throughput vs concurrency; ITL distribution; £/1M tokens per config)
    - Files: `results/*/charts/`
    - Verify: generated by the script, not by hand.
- [x] `[MUST]` **7.3 Generated report** (`make report`; regenerates byte for byte, tested)
    - Expected result: per-run `summary.md` and `docs/report.md` generated from results, covering methodology, hardware, limitations and the max load meeting the SLO per config.
    - Verify: deleting and regenerating reproduces the same report.
- [x] `[MUST]` **7.4 CI (GitHub Actions)** (`checks` + `stack` jobs; 0.5B model cached; rehearsed locally)
    - Expected result: lint + unit + contract; integration job with the 0.5B GGUF model in Compose (model cached).
    - Files: `.github/workflows/ci.yml`
    - Verify: green badge on GitHub.
- [ ] `[MUST]` **7.5 Final README**
    - Expected result: architecture diagram, generated results table, quick start verified from a clean clone, the "in production I'd use LiteLLM" note, limitations (free GPU class, assumed prices).
    - Verify: a fresh clone runs in under 10 minutes following the README.
- [ ] `[MUST]` **7.6 ADRs 001–003 finalised and indexed**
- [ ] `[SHOULD]` **7.7 3-minute demo video** (load test with the dashboard moving and a backend killed)
- [ ] `[SHOULD]` **7.8 CV bullets and interview notes** filled with real numbers only
    - Files: `docs/interview-notes.md`
- [ ] `[OPTIONAL]` **7.9 Sizing calculator** (cheapest measured config meeting a given SLO)
- [ ] `[OPTIONAL]` **7.10 Advanced:** SLO-aware routing on live queue depth; speculative decoding / prefix-cache experiments; SGLang; Kubernetes + KEDA

---

## Definition of done (MVP)

- [ ] Streaming OpenAI-compatible gateway, proven by OpenAI SDK contract tests
- [ ] ≥ 2 backends behind one abstraction, with fallback
- [ ] API keys, tenants, model policy, Redis rate limits, Postgres usage ledger with £
- [ ] Timeouts, pre-first-token retries, circuit breaker, graceful shutdown
- [ ] JSON logs, Prometheus metrics, Grafana dashboard, OTel traces
- [ ] Benchmark of ≥ 3 real configurations with TTFT, ITL, p50/p95/p99, throughput, gateway overhead, £/1M tokens, saved as Parquet
- [ ] Tests (unit/contract/integration), CI, 3 ADRs, generated report, professional README

## Progress log

| Date | Day | Done | Notes / schedule changes |
|---|---|---|---|
| 2026-10-08 | 0 | Planning docs | Environment inspected; deviations proposed |
| 2026-10-08 | 1 | 1.1 brief, 1.2 git + GitHub | Considered dropping `make`; kept it after finding the cause was a stale VS Code PATH (see development.md §10). 1.3 now also checks the WSL2 backend. |
| 2026-10-08 | 1 | 1.3–1.6, 1.8–1.10 | **Day 1 checkpoint met:** OpenAI SDK → Relay → llama.cpp (CPU, non-streaming), verified from a clean clone. Relay moved to port 8000 (8080 taken by another project). 1.7 (SHOULD) was blocked on the NVIDIA driver; driver updated to 617.42 and GPU service added (`make up-gpu`, :8082). llama.cpp ports bound to 127.0.0.1. **Day 1 complete.** |
| 2026-10-10 | 2 | 2.1–2.10 | **Day 2 complete.** Streaming gateway with backend abstraction, YAML routing, `/v1/models`, usage capture, OpenAI-format errors, request limits; 111 unit + 7 contract tests. Order changed to do streaming before the abstraction. 2.9 retagged SHOULD → MUST (spec requires size limits). Colab check passed (T4, vLLM 0.31.0); findings: `--dtype half`, port clash with Relay, Colab on Python 3.13. 2.11 (OPTIONAL) skipped. |
| 2026-10-11 | 3 | 3.1–3.7 | **Day 3 complete; checkpoint met live:** `batch-summariser` (30/min) sent 40 concurrent requests → 30×200, 10×429 (`retry-after: 2`) while `support-bot` (60/min) got 10×200; ledger shows 30 and 10 rows. Postgres on host port 5433 (native PG18 on 5432). Decisions: tenant policy in Postgres; forbidden model → 404 like unknown (was 403); omitted `max_tokens` gets the tenant cap; ADR-002 fail open to local limits; `cost_gbp` NULL until benchmark rates. 3.8 (OPTIONAL) skipped. 164 unit, 20 integration, 8 contract tests. |
| 2026-10-13 | 4 | 4.1–4.12 | **Day 4 complete.** 7 commits: timeouts + retries before the first token (ADR-003); circuit breakers + fallback routing; graceful shutdown + `/readyz` (`make serve`); JSON logs + request ids + Prometheus metrics; Prometheus/Grafana (port 3001) + 7 SLO alert rules with promtool tests; OpenTelemetry → Jaeger; failure-scenario tests + Relay Dockerfile (`make up-all`). Checkpoint met live: stopping the GPU primary under traffic opened its breaker (`relay_circuit_breaker_state=2`), traffic fell back to CPU, the trace showed the retries. Notes: on Windows a refused connection takes ~2 s (breaker limits the cost); ports 8000 and 3000 held by other local projects (checks used 8010; Grafana on 3001). 213 unit, 26 integration, 8 contract tests. |
| 2026-10-16 | 5 | 5.1–5.9, 5.11 | **Day 5 complete; checkpoint met:** `make bench SCENARIO=bench/scenarios/local-smoke.yaml` produces a full results folder (config, Parquet per request, summary, env). Harness: seeded scenarios, open loop (Poisson) and closed loop, TTFT/ITL/TPOT/E2E/goodput, results + DuckDB, cost model (`bench/pricing.yaml`, sourced prices, RTX 4050 TBD). Validated against a fake engine's recorded timeline (scripted delays are not ground truth: Windows sleeps snap to ~15.6 ms). `bench.overhead` measures Relay's added latency per request (smoke: ~4 ms p50 before the first token, agrees with the server-side histogram; p99 ~32 ms tail to investigate). **Found in the dry run:** llama.cpp reuses cached prompt prefixes across requests, so levels/runs resending the same seeded prompts got TTFT 53-102 ms instead of 585-1107 ms; prompt openings are now salted per run and level (`prompt_salt` in env.json). Timed runs need the laptop awake on mains power (a run failed in Modern Standby). 5.10 GSM8K (SHOULD) moved to Day 6. 250 unit, 28 integration, 8 contract tests. |
| 2026-10-16/17 | 6 | 6.1 | **Day 6 in progress.** ADR-001 matrix: local RTX 4050 llama.cpp (CPU baseline, 1.5B Q4_K_M, 1.5B Q8_0, 7B Q4_K_M) and Colab T4 vLLM (1.5B FP16, 1.5B AWQ, 7B AWQ). Benchmark setup: single-backend Relay configs without fallback or retries, `bench` tenant (key in `.env`), 4 slots x 1024 tokens, scenario `setup` metadata, `--probe`, lean stack (`make bench-stack-gpu/-cpu`), memory guard, invalid runs to `results/_invalid/`. Found and fixed: F7 llama.cpp drops reused connections (Relay now opens a new connection per request to llama.cpp; retries had hidden it), F8 RAM exhaustion stalled everything and hung Docker, F9 llama.cpp's 8 GiB host-RAM prompt cache (`--cache-ram 0` for benchmarks). First full GPU 1.5B Q4_K_M run flagged invalid (Chrome grew to 2.3 GB, RAM fell to 0.81 GiB): no local results yet. Open question: 2 workers barely faster than 1 (F9). 7B download failed (HF network error), not retried. **Next:** 6.3 Colab notebook + 6.4 runs; then local runs in one sitting with the browser closed. 275 unit, 31 integration, 8 contract tests. |
| 2026-10-22/25 | 6 | 6.3-6.5 | **Day 6 complete (Colab).** Relay standalone mode (no Postgres/Redis) for Colab; notebook `bench/colab/relay_benchmark.ipynb` + `scripts/colab_run.sh` (dry-run in Linux with a stub vLLM); harness `--auto-rates`, client-lag and memory validity checks. Results on a Colab T4, vLLM 0.31.0 (F10-F12): SLO limit 1.5B AWQ 6.36 req/s (~GBP 0.19/1M output tokens), 1.5B FP16 5.14 req/s (~GBP 0.24), 7B AWQ 0.84 req/s (~GBP 1.41); GSM8K 65.5% / 72.0% / 95.0% (AWQ vs FP16 p = 0.066). Probe widened to 64 workers after the first 1.5B runs never saturated. **Deviation:** local RTX 4050 runs (6.2) dropped, laptop an unreliable benchmark host (architecture.md §6, ADR-001 amendment). 6.6 moved to Day 7. 300 unit, 32 integration, 8 contract tests. |
