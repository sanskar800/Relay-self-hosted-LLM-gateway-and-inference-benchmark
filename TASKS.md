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
- [ ] `[SHOULD]` **1.7 llama.cpp on the RTX 4050 (CUDA image)**
    - Why: the realistic local GPU path; needed for Day 6.
    - Expected result: a `llamacpp-gpu` Compose profile with GPU device reservation and `--n-gpu-layers` set.
    - Files: `deploy/compose/docker-compose.yml`
    - Verify: `nvidia-smi` shows the container using VRAM; server logs show layers offloaded to CUDA. (Only an observation, **not** a benchmark number.)
- [ ] `[MUST]` **1.8 Minimal Relay: FastAPI app with non-streaming proxy**
    - Why: the smallest possible gateway, so every later feature has something to attach to.
    - Expected result: `GET /healthz`; `POST /v1/chat/completions` forwards the JSON body to llama.cpp with a shared `httpx.AsyncClient` (created in the app lifespan) and returns the response.
    - Files: `src/relay/main.py`, `src/relay/config.py`, `.env.example`
    - Verify: `curl :8080/v1/chat/completions` returns the model's answer.
- [ ] `[MUST]` **1.9 End-to-end with the official OpenAI SDK**
    - Why: compatibility should be proven, not assumed. This becomes the first contract test.
    - Expected result: `tests/contract/test_openai_sdk.py` uses `OpenAI(base_url="http://localhost:8080/v1")`.
    - Files: `tests/contract/`
    - Verify: `uv run pytest tests/contract` passes with the stack running.
- [ ] `[MUST]` **1.10 Update docs and commit**
    - Expected result: quick start in README and development.md actually works from a clean checkout.
    - Verify: follow the README in a fresh terminal.

**End of Day 1 checkpoint:** OpenAI SDK → Relay → llama.cpp works (non-streaming).

---

## Day 2: Gateway (≈5 h)

Goal: **a correct streaming, OpenAI-compatible gateway with a backend abstraction.**

- [ ] `[MUST]` **2.1 OpenAI request/response schemas (Pydantic)**
    - Why: validate input at the edge; reject bad requests before they reach a GPU.
    - Expected result: models for `ChatCompletionRequest` (the subset we support, with unknown fields passed through), responses and error objects.
    - Files: `src/relay/api/schemas.py`
    - Verify: unit tests for valid, invalid and extra-field payloads.
- [ ] `[MUST]` **2.2 SSE streaming passthrough**
    - Why: streaming is what makes LLM UX tolerable, and it is the hardest part of a proxy to get right (flushing, backpressure, disconnects).
    - Expected result: `stream: true` uses `httpx` `client.stream()` and FastAPI `StreamingResponse`; chunks are forwarded as they arrive; client disconnect cancels the upstream request.
    - Files: `src/relay/api/chat.py`
    - Verify: `curl -N` shows tokens arriving incrementally; the contract test with `stream=True` passes; disconnect test: llama.cpp logs show the request cancelled.
- [ ] `[MUST]` **2.3 Backend abstraction**
    - Why: routing, retries and accounting must not depend on which engine is behind them.
    - Expected result: `Backend` protocol (`chat`, `stream`, `health`, `name`); `OpenAICompatBackend` base; `LlamaCppBackend`, `VLLMBackend` subclasses.
    - Files: `src/relay/backends/{base,openai_compat,llamacpp,vllm}.py`
    - Verify: unit tests with a fake backend.
- [ ] `[MUST]` **2.4 Model/backends config (YAML)**
    - Why: the public model name (`qwen2.5-1.5b-instruct`) maps to an ordered list of backends. This is the basis for routing and fallback.
    - Expected result: `config/relay.yaml` loaded and validated at startup.
    - Files: `config/relay.yaml`, `src/relay/config.py`
    - Verify: startup fails with a clear error on invalid config (tested).
- [ ] `[MUST]` **2.5 `GET /v1/models`**
    - Why: SDKs and tools call it; part of API compatibility.
    - Files: `src/relay/api/models.py`
    - Verify: `client.models.list()` in the contract tests.
- [ ] `[MUST]` **2.6 Error handling in OpenAI format**
    - Why: clients parse `{"error": {...}}`; an upstream 500 should not leak as an HTML page.
    - Expected result: exception handlers; upstream errors mapped (502/503/504); a mid-stream failure emits an SSE error event and closes cleanly.
    - Files: `src/relay/api/errors.py`
    - Verify: unit tests; the SDK raises the right `openai.APIError` subclass.
- [ ] `[MUST]` **2.7 Token usage capture**
    - Why: accounting (Day 3) needs prompt and completion tokens for both streaming and non-streaming requests.
    - Expected result: usage read from the response; for streams, request `stream_options.include_usage` from the backend, with a documented fallback if the engine does not support it.
    - Files: `src/relay/accounting/usage.py`
    - Verify: tests assert the token counts match the engine's report.
- [ ] `[MUST]` **2.8 Unit tests with a fake backend** (no model needed)
    - Why: fast, deterministic tests that do not need a running model.
    - Expected result: a fake OpenAI-compatible server (respx or an in-process ASGI app) that streams scripted chunks.
    - Files: `tests/unit/`, `tests/conftest.py`
    - Verify: `uv run pytest tests/unit` runs in under 5 s.
- [ ] `[SHOULD]` **2.9 Request size and parameter limits** (body bytes, `max_tokens` ceiling, message count)
    - Files: `src/relay/api/limits.py`
    - Verify: oversized request gets 413/400.
- [ ] `[MUST]` **2.10 Colab feasibility check (≈20 min, de-risking only, no benchmarking)**
    - Why: the benchmark depends on a free Colab GPU running vLLM. Finding out on Day 6 that it doesn't work would leave no time to fix it.
    - Expected result: in a scratch Colab notebook, the runtime shows a GPU (`nvidia-smi`); `pip install vllm` succeeds; `vllm serve Qwen/Qwen2.5-1.5B-Instruct` answers one curl request. Record the GPU model, vLLM version and any workarounds in development.md.
    - Files: `docs/development.md`
    - Verify: one successful completion from vLLM on Colab. If it fails, revise the Day 6 plan now.
- [ ] `[OPTIONAL]` **2.11 `/v1/completions` and `/v1/embeddings` passthrough**

**Checkpoint:** streaming works through Relay via the OpenAI SDK; unit tests green.

---

## Day 3: Multi-tenancy and governance (≈5 h)

Goal: **every request is attributed to a tenant, limited and recorded.**

- [ ] `[MUST]` **3.1 PostgreSQL + Redis in Compose; schema**
    - Why: Postgres is the durable ledger and key store; Redis is the shared rate-limit state.
    - Expected result: services with health checks; `deploy/compose/initdb/001_schema.sql` (tables `tenants`, `api_keys(key_hash, tenant_id, …)`, `usage_events`). Plain SQL, no migration framework (deliberate simplicity).
    - Files: `deploy/compose/`, `src/relay/accounting/db.py`
    - Verify: `psql` shows the tables; `redis-cli ping` returns PONG.
- [ ] `[MUST]` **3.2 API keys and tenant identification**
    - Why: no identity means no quotas and no cost attribution.
    - Expected result: `Authorization: Bearer rk_…`; SHA-256 hash lookup with a short in-memory TTL cache; 401 in OpenAI error format; admin CLI `relay-admin create-tenant / create-key` (shows the key once).
    - Files: `src/relay/auth/`, `src/relay/cli.py`
    - Verify: tests for valid, invalid and revoked keys; the raw key never appears in the DB or logs.
- [ ] `[MUST]` **3.3 Per-tenant model policy**
    - Why: governance. Team A may use model X but not model Y, and has a per-request `max_tokens` cap.
    - Expected result: allow-list and caps checked before routing; 403 if not allowed.
    - Files: `src/relay/auth/policy.py`
    - Verify: unit tests.
- [ ] `[MUST]` **3.4 Redis token-bucket rate limiter (Lua)**
    - Why: fair sharing of a scarce GPU across tenants and replicas; the Lua script makes "refill + check + take" atomic.
    - Expected result: request/min and token/min buckets per tenant; 429 with `Retry-After` and `x-ratelimit-*` headers.
    - Files: `src/relay/limits/token_bucket.py`, `src/relay/limits/token_bucket.lua`
    - Verify: unit tests of the refill maths (fake clock); integration test against real Redis: N+1 rapid requests and the last one is 429.
- [ ] `[MUST]` **3.5 ADR-002: limiter behaviour when Redis is down, and its implementation**
    - Why: a classic interview question. Fail-open keeps an internal tool available; fail-closed protects a paid API.
    - Expected result: ADR written; the chosen behaviour implemented (likely fail-open to a local in-memory bucket, plus a metric and a warning log).
    - Files: `docs/adr/002-rate-limit-failure-behaviour.md`, `src/relay/limits/`
    - Verify: integration test that stops Redis, sends requests and asserts the documented behaviour.
- [ ] `[MUST]` **3.6 Non-blocking usage ledger**
    - Why: cost attribution must be durable, but a slow database must never slow a user's request.
    - Expected result: one usage event per request (tenant, key ID, model, backend, prompt/completion tokens, TTFT, duration, status, estimated £); bounded `asyncio.Queue` and a batch writer task; buffering when Postgres is down; dropped events counted.
    - Files: `src/relay/accounting/ledger.py`, `config/pricing.yaml`
    - Verify: after N requests, `SELECT count(*)` equals N; Postgres-down test shows requests still succeed.
- [ ] `[SHOULD]` **3.7 Usage report query**
    - Expected result: SQL view `usage_daily` (tenant, day, tokens, £) and `relay-admin usage` CLI.
    - Verify: matches what the test traffic generated.
- [ ] `[OPTIONAL]` **3.8 Exact-match response cache** (temperature = 0 only)

**Checkpoint:** two tenants with different limits; one gets 429 while the other is unaffected; the ledger shows both.

---

## Day 4: Reliability and observability (≈5–6 h, the heaviest day)

Goal: **Relay survives backend failure and you can see what it is doing.**

- [ ] `[MUST]` **4.1 Per-backend timeouts** (connect, first byte, idle between chunks, total)
    - Why: a hung backend must not hold client connections forever.
    - Files: `src/relay/backends/base.py`, `config/relay.yaml`
    - Verify: a fake slow backend triggers 504 within the configured time.
- [ ] `[MUST]` **4.2 Retries only before the first token (ADR-003)**
    - Why: once tokens reach the user, a retry would produce duplicated or contradictory text.
    - Expected result: retry with exponential backoff + jitter on connect errors / 503 only, before any byte is sent; mid-stream failure leads to an SSE error event and no retry.
    - Files: `src/relay/routing/retry.py`, `docs/adr/003-streaming-retry-policy.md`
    - Verify: tests "fails before first token, so retried" and "fails after 3 tokens, so clean error, no retry".
- [ ] `[MUST]` **4.3 Circuit breaker**
    - Why: stop sending traffic to a backend that is clearly down; probe it to recover.
    - Expected result: closed → open → half-open state machine per backend, with configurable thresholds.
    - Files: `src/relay/routing/circuit_breaker.py`
    - Verify: unit tests drive every state transition with a fake clock.
- [ ] `[MUST]` **4.4 Routing and fallback**
    - Why: availability. If the primary is down, use the next backend for that model.
    - Expected result: the router picks the first backend whose breaker is not open; falls back on pre-first-token failure. Local demo: llama.cpp GPU primary, llama.cpp CPU secondary.
    - Files: `src/relay/routing/router.py`
    - Verify: `docker stop` the primary; requests still succeed via the secondary; the breaker metric shows open.
- [ ] `[MUST]` **4.5 Graceful shutdown**
    - Why: deploys should not cut users off mid-answer.
    - Expected result: on SIGTERM, `/readyz` returns 503, new requests are rejected and in-flight streams drain up to a deadline.
    - Files: `src/relay/main.py`
    - Verify: start a long stream, send SIGTERM, and the stream completes.
- [ ] `[MUST]` **4.6 Structured JSON logs**
    - Expected result: structlog JSON with request ID, tenant, model, backend, status and durations; **no prompt bodies** by default.
    - Files: `src/relay/telemetry/logging.py`
    - Verify: log line inspection; a test asserts the prompt text is absent.
- [ ] `[MUST]` **4.7 Prometheus metrics**
    - Why: latency percentiles and per-tenant usage are the product.
    - Expected result: `/metrics` with request count by status/backend, TTFT and ITL histograms, output tokens, gateway overhead histogram, in-flight per backend, breaker state gauge, tokens and £ per tenant, ledger drops.
    - Files: `src/relay/telemetry/metrics.py`
    - Verify: `curl :8080/metrics` after traffic.
- [ ] `[MUST]` **4.8 Prometheus + Grafana in Compose with a provisioned dashboard**
    - Expected result: dashboard JSON committed: traffic, errors, TTFT/ITL p50/p95/p99, tokens/s, per-tenant usage, breaker state.
    - Files: `monitoring/prometheus.yml`, `monitoring/grafana/`
    - Verify: `make up`, generate traffic, panels move.
- [ ] `[MUST]` **4.9 Basic OpenTelemetry tracing → Jaeger**
    - Why: shows where time goes inside a single request.
    - Expected result: FastAPI + httpx auto-instrumentation plus manual spans for auth, limit and route; exported via OTLP to Jaeger in Compose.
    - Files: `src/relay/telemetry/tracing.py`, compose
    - Verify: a trace in the Jaeger UI shows the stage spans.
- [ ] `[MUST]` **4.10 Failure-scenario integration tests**
    - Expected result: backend killed mid-stream; slow backend → timeout → fallback; Redis down → documented behaviour; Postgres down → requests still served.
    - Files: `tests/integration/`
    - Verify: `make test-integration` is green.
- [ ] `[SHOULD]` **4.11 Relay Dockerfile; `make up` runs the whole stack**
    - Files: `Dockerfile`, compose
    - Verify: fresh clone, then `make up`, then curl works.
- [ ] `[SHOULD]` **4.12 Prometheus alert rules** (error rate, TTFT p95 above SLO)
    - Files: `monitoring/rules.yml`

**Checkpoint:** kill a backend under traffic; the dashboard shows the breaker opening and fallback taking over.

---

## Day 5: Benchmark harness (≈5 h)

Goal: **a trustworthy measuring instrument, validated before it measures anything real.**

- [ ] `[MUST]` **5.1 Scenario config schema (YAML + Pydantic)**
    - Expected result: target URL, model, arrival rates (req/s), duration, warm-up, prompt/output length distribution, seed, SLO.
    - Files: `bench/scenarios/*.yaml`, `bench/loadgen/config.py`
    - Verify: invalid configs are rejected.
- [ ] `[MUST]` **5.2 Prompt/length mix**
    - Why: benchmarks with one fixed prompt length flatter the engine.
    - Expected result: seeded synthetic mix with a documented length distribution. `[SHOULD]`: lengths sampled from a public conversation dataset.
    - Files: `bench/loadgen/workload.py`
    - Verify: a histogram of generated lengths matches the documented distribution.
- [ ] `[MUST]` **5.3 Open-loop async load generator (Poisson arrivals)**
    - Why: closed-loop testing hides overload because the client slows down with the server.
    - Expected result: arrivals scheduled independently of completions; per-request timestamps (send, first token, every token, end), token counts, status.
    - Files: `bench/loadgen/runner.py`
    - Verify: at a fixed rate λ, the observed mean inter-arrival time ≈ 1/λ.
- [ ] `[MUST]` **5.4 Metric computation**
    - Expected result: TTFT, ITL, TPOT, E2E latency, request throughput, output tokens/s, goodput (share meeting the SLO), error rate; p50/p95/p99.
    - Files: `bench/loadgen/metrics.py`
    - Verify: unit tests on hand-built timelines with known answers.
- [ ] `[MUST]` **5.5 Harness self-validation against a fake backend with known latency**
    - Why: proves the instrument is accurate before trusting it on a real model.
    - Expected result: a fake server with injected TTFT/ITL; the harness recovers those values within tolerance.
    - Files: `tests/integration/test_harness_accuracy.py`
    - Verify: test passes.
- [ ] `[MUST]` **5.6 Gateway overhead measurement**
    - Expected result: the same scenario run direct-to-backend and via Relay; plus a server-side overhead histogram.
    - Verify: both paths produce results; the difference is reported (value TBD until measured).
- [ ] `[MUST]` **5.7 Results storage**
    - Expected result: `results/run-YYYY-MM-DD-<name>/` containing `config.yaml`, `results.parquet` (one row per request), `env.json` (GPU, driver, engine version, git SHA).
    - Files: `bench/loadgen/results.py`
    - Verify: `duckdb -c "select count(*) from 'results/*/results.parquet'"`.
- [ ] `[MUST]` **5.8 Cost model**
    - Expected result: `£/1M output tokens` from measured tokens/s and `bench/pricing.yaml` (GPU class, £/h, source URL, date). Prices are labelled as assumptions.
    - Files: `bench/analysis/cost.py`, `bench/pricing.yaml`
    - Verify: unit test of the formula.
- [ ] `[MUST]` **5.9 Local dry run** (1.5B model, llama.cpp, low rates)
    - Verify: a complete results folder is produced end to end. Treat this as a smoke test, not headline numbers.
- [ ] `[SHOULD]` **5.10 GSM8K quality evaluation** (fixed 200-question subset, exact match, temperature 0, via Relay)
    - Files: `bench/quality/gsm8k.py`
    - Verify: runs on the dev model; accuracy recorded only from the actual run.
- [ ] `[SHOULD]` **5.11 Closed-loop mode** (to demonstrate the overload-hiding effect in the report)

**Checkpoint:** `make bench SCENARIO=bench/scenarios/local-smoke.yaml` produces a valid results folder.

---

## Day 6: GPU benchmark (≈5 h, £0)

Goal: **real measurements on hardware we actually have.** We only compare configurations that actually ran.

- [ ] `[MUST]` **6.1 Fix the final configuration matrix and write ADR-001**
    - Why: decide what is feasible on a 6 GB RTX 4050 and the free Colab GPU before spending time.
    - Candidate matrix (to confirm):
        - Local RTX 4050: llama.cpp CUDA, Qwen2.5-1.5B Q4_K_M vs Q8_0; 7B Q4_K_M if it fits. Compare with the llama.cpp CPU baseline.
        - Colab GPU: vLLM, Qwen2.5-1.5B FP16 vs AWQ; 7B AWQ; `max_num_seqs` sweep; prefix caching on/off `[SHOULD]`.
    - Files: `docs/adr/001-inference-engines-and-hardware.md`, `bench/scenarios/`
    - Verify: each config has a scenario file.
- [ ] `[MUST]` **6.2 Local GPU benchmark runs**
    - Expected result: results folders for each local config across several arrival rates.
    - Verify: the Parquet row count matches requests sent; error rate recorded.
- [ ] `[MUST]` **6.3 Colab notebook**
    - Expected result: `bench/colab/relay_benchmark.ipynb` does: clone repo → `uv`/pip install → detect GPU → start vLLM (and Relay in minimal mode: in-memory limiter, ledger off) → run scenarios → zip `results/` → download or save to Drive. No code edits inside Colab.
    - Files: `bench/colab/`, `scripts/colab_run.sh`
    - Verify: runs top to bottom on a fresh runtime.
- [ ] `[MUST]` **6.4 Colab vLLM runs**
    - Expected result: results for at least 3 engine/quantisation configs (MVP requirement), with GPU model recorded in `env.json`.
    - Verify: results are committed under `results/`.
- [ ] `[SHOULD]` **6.5 GSM8K per quantisation config**
- [ ] `[SHOULD]` **6.6 Grafana screenshot under load** (local stack)
- [ ] `[OPTIONAL]` **6.7 vLLM locally on the RTX 4050 via WSL/Docker** (small model; FP8 is possible on Ada)

**Known risks:** free Colab GPUs are not guaranteed (retry later or use a different time of day); sessions can disconnect, so save results after each scenario; recent vLLM releases may drop support for older GPUs (check compute capability 7.5 support; pin a version if needed).

---

## Day 7: Analysis and portfolio polish (≈5 h)

- [ ] `[MUST]` **7.1 DuckDB analysis over all runs**
    - Files: `bench/analysis/queries.sql`, `bench/analysis/analyze.py`
    - Verify: the script regenerates every table from Parquet.
- [ ] `[MUST]` **7.2 Charts** (TTFT p95 vs arrival rate; throughput vs concurrency; ITL distribution; £/1M tokens per config)
    - Files: `results/*/charts/`
    - Verify: generated by the script, not by hand.
- [ ] `[MUST]` **7.3 Generated report**
    - Expected result: per-run `summary.md` and `docs/report.md` generated from results, covering methodology, hardware, limitations and the max load meeting the SLO per config.
    - Verify: deleting and regenerating reproduces the same report.
- [ ] `[MUST]` **7.4 CI (GitHub Actions)**
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
