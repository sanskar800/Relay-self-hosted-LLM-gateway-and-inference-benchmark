# Development guide

A living setup guide. Each external requirement lists **what it is, why we need it, how to check for it, how to install it and how to verify it.** Update this file whenever setup changes.

## 0. Machine inventory (checked 2026-10-08)

| Item | Found | Notes |
|---|---|---|
| OS | Windows 11 Pro | WSL2 with Ubuntu (Python 3.12.3 inside) |
| CPU | Intel i7-14650HX, 16 cores / 24 threads | Strong CPU baseline for llama.cpp |
| RAM | ~16 GB | Limits how many containers and models run at once |
| GPU | NVIDIA RTX 4050 Laptop, **6 GB** VRAM, compute capability 8.9 (Ada) | ~0.5 GB already used by the desktop |
| NVIDIA driver | **617.42 (CUDA 13.4)**, updated 2026-10-08 from 552.27 (CUDA 12.4) | Update needed for llama.cpp CUDA images (≥ 12.8) |
| Docker | Docker Desktop 29.1.3, Compose v2.40.3 | `nvidia` runtime is registered |
| Python | 3.13.7 (Windows) | We will use a uv-managed 3.12 (see below) |
| uv | 0.9.9 | ✅ |
| Git | 2.51.0 | ✅ |
| make | GNU Make 4.4.1 (installed 2026-10-08 via winget) | ✅ |
| Disk | ~207 GB free on C: | Plenty for models |

### What 6 GB of VRAM means in practice

| Model (GGUF, approximate weight size) | Fits on the RTX 4050? |
|---|---|
| Qwen2.5-0.5B Q4_K_M (~0.4 GB) | Easily. Use it for CI and tests. |
| Qwen2.5-1.5B Q4_K_M (~1 GB) / Q8_0 (~1.7 GB) | Easily. Use it for **daily development**. |
| 7B Q4_K_M (~4.7 GB) | Probably, with small context and few parallel slots. Benchmark only, to be verified. |
| 7B FP16 (~15 GB) | No. |

## 1. Repository location

**Decision (2026-10-08):** the project lives at **`C:\dev\5projects\relay`**, outside OneDrive. GitHub is the backup: https://github.com/sanskar800/Relay-self-hosted-LLM-gateway-and-inference-benchmark

Why not OneDrive:

1. OneDrive tries to sync `.venv/` (thousands of files) and downloaded models (GBs).
2. File locking during sync can break `git` and `uv`.
3. Docker bind mounts from OneDrive paths are slow.

Models are stored **outside** the repository as well (see §5).

## 2. Tools

### uv (Python package and project manager)
- **What/why:** it creates the virtual environment, installs dependencies from `pyproject.toml`, writes a lock file (`uv.lock`) for reproducible installs, and downloads the right Python version automatically.
- **Check:** `uv --version` ✅ installed.
- **Python version:** we pin **3.12** with a `.python-version` file. uv downloads 3.12 on first `uv sync`, locally and in Colab, so the project's interpreter does not depend on the machine's. *Correction (2026-10-10):* the original reason, "Colab runs 3.12", is out of date: the Colab check found Python 3.13.15. Moving the pin to 3.13 is possible but undecided.

### Docker Desktop + Compose
- **What/why:** runs Redis, PostgreSQL, llama.cpp, Prometheus, Grafana and Jaeger without installing them on Windows.
- **Check:** `docker --version` and `docker compose version` ✅.
- **GPU in Docker (Day 1 check):**
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  It should print the RTX 4050. If it does, the llama.cpp CUDA container can use the GPU.
- **Result (2026-10-08, task 1.3):** ✅
  | Check | Result |
  |---|---|
  | Docker backend | WSL2 (`docker info` kernel `6.6.87.2-microsoft-standard-WSL2`; `wsl -l -v` shows `docker-desktop` v2) |
  | Runtimes | `nvidia`, `runc` (default `runc`; `--gpus all` selects the GPU) |
  | Docker VM resources | 24 CPUs, ~8.2 GB RAM (WSL2 default: half of host RAM) |
  | GPU in container | `NVIDIA GeForce RTX 4050 Laptop GPU`, 6141 MiB, driver 552.27, **max CUDA 12.4** |
- **Notes for later tasks:**
  - Under WSL2, `nvidia-smi` shows **no per-process list**, even when a container uses the GPU. Confirm GPU use by the **Memory-Usage** total (≈ 258 MiB at idle) and the engine's own logs.
  - A container's CUDA version must be **≤ 12.4** with this driver. If a llama.cpp CUDA image needs newer CUDA, either update the NVIDIA driver (free) or build against CUDA 12.4 (checked in task 1.7).
- **llama.cpp CUDA images vs this driver (checked 2026-10-08):** every `server-cuda*-b11459` image (`cuda`, `cuda12`, `cuda13`) declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8` (13.4 for `cuda13`), read without pulling via `docker buildx imagetools inspect <image> --format '{{json .Image}}'`.
  - Default run: `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8, please update your driver to a newer version`.
  - With `-e NVIDIA_DISABLE_REQUIRE=1` (skip the check, relying on CUDA minor-version compatibility): container starts but `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`. Not viable.
  - **Conclusion:** GPU llama.cpp in Docker needs an NVIDIA driver that supports CUDA ≥ 12.8 (driver ≥ 570). **Resolved 2026-10-08:** driver updated to 617.42 (CUDA 13.4); `nvidia-smi` now labels this "CUDA UMD Version". Alternatives rejected: building llama.cpp against CUDA 12.4 (long compile, image to maintain); an older llama.cpp build (engine version would differ from the CPU baseline).
  - ~8 GB for the whole Compose stack (models + Postgres + Redis + Grafana + …). If memory gets tight, raise it in `%USERPROFILE%\.wslconfig` (`[wsl2] memory=10GB`).

### make
- **What/why:** short, memorable commands (`make up`, `make test`, `make bench`) that work the same on Linux, macOS, WSL, CI and Colab. All five portfolio repos will share this convention.
- **Check:** `make --version` → GNU Make 4.4.1 ✅ (installed 2026-10-08).
- **Targets:** `make help` lists them (`setup`, `test`, `lint`, `format`, `models`, `models-all`; more are added as the stack grows).
- **Install (Windows):** `winget install ezwinports.make`, then open a **new** terminal so PATH refreshes.
- **Verify:** `make --version`.
- **Alternative:** run the same targets from WSL Ubuntu (`sudo apt install make`).

### Git + GitHub
- Git ✅. GitHub remote creation is a Day 1 task.

## 3. Python project

```bash
uv sync                    # create .venv (Python 3.12) and install exact versions from uv.lock
uv run pytest              # run tests (unit tests need no model or Docker)
uv run ruff check .        # lint
uv run ruff format .       # format (CI runs `ruff format --check .`)
```

- **Layout:** code in `src/relay/` (src layout, so tests import the *installed* package, not the source folder); tests in `tests/{unit,contract,integration}/`.
- **Dependencies:** runtime = FastAPI, httpx, pydantic-settings, uvicorn. Dev only = pytest, pytest-asyncio, ruff, **openai** (the SDK is used only by contract tests as a client; Relay never imports it).
- **Adding a dependency:** `uv add <pkg>` (or `uv add --dev <pkg>`). This updates `pyproject.toml` and `uv.lock`; commit both.
- **Test markers:** `contract` (needs running stack), `integration` (needs Compose services). Run a subset with `uv run pytest -m "not contract and not integration"`.
- **Verified 2026-10-08:** Python 3.12.12, 1 test passed, ruff clean.

## 4. Environment variables

*(filled in as settings are added; the source of truth is `.env.example`)*

| Variable | Purpose | Default |
|---|---|---|
| `RELAY_CONFIG` | Path to backends/models YAML | `config/relay.yaml` |
| `RELAY_REDIS_URL` | Rate-limit store | `redis://127.0.0.1:6379/0` |
| `RELAY_DATABASE_URL` | Tenants, API keys, usage ledger | `postgresql://relay:relay@127.0.0.1:5433/relay` (host port **5433**) |

`.env` (git-ignored) is created from `.env.example`: `cp .env.example .env`. The database credentials are **local development values** (loopback-bound container, used nowhere else); a real deployment would take them from a secret store and give Redis a password.
| `HF_TOKEN` | Only for gated Hugging Face models (not needed for Qwen2.5) | unset |

## 5. Models

- Models live **outside** the repo, in `%USERPROFILE%\models` (Windows) or `~/models` (WSL/Colab), mounted read-only into containers. Override with `RELAY_MODELS_DIR`.
- `make models` (= `uv run python scripts/download_models.py`) downloads them with `huggingface_hub`. Each file is pinned by repo, filename, **revision** (Hugging Face commit) and **SHA-256**; the script rejects a file whose size or hash differs. Re-running only re-verifies.
- No Hugging Face login needed (repos are not gated). The "unauthenticated requests" warning is harmless.

| Purpose | Repository | File | Size | Downloaded by |
|---|---|---|---|---|
| CI / tests | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ `9217f5d` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 0.49 GB | `make models` |
| Development | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` @ `91cad51` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.12 GB | `make models` |
| Benchmark (Q8 vs Q4) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` @ `91cad51` | `qwen2.5-1.5b-instruct-q8_0.gguf` | 1.89 GB | `make models-all` |

- **Licence:** Apache-2.0 for all three (model card metadata, checked 2026-10-08).
- **Verified 2026-10-08:** CI + dev models downloaded in ~2.5 min; SHA-256 matched; re-run skipped the download.

## 6. Local stack

```bash
make up      # docker compose -f deploy/compose/docker-compose.yml up -d --wait  (returns once healthy)
make ps      # status
make logs    # follow logs
make down    # stop
```

| Service | Port (host) | Image | Notes |
|---|---|---|---|
| `llamacpp` | 8081 | `ghcr.io/ggml-org/llama.cpp:server-b11459` (0.6.0-dev, commit `f498f864f`) | CPU; 2 slots × 4096 ctx; `--metrics`; model via `LLAMACPP_MODEL`, models dir via `RELAY_MODELS_DIR` |
| `llamacpp-gpu` | 8082 | `ghcr.io/ggml-org/llama.cpp:server-cuda12-b11459` (same build, CUDA) | **Profile `gpu`**: `make up-gpu`. All layers on the RTX 4050 (`--n-gpu-layers 99`); model via `LLAMACPP_GPU_MODEL` |

Both ports are bound to **127.0.0.1**: llama.cpp has no authentication (its log warns `no API key is set and CORS allows all origins`), so it must not be reachable from the network; clients go through Relay.

**Confirming GPU use** (2026-10-08; observations, not benchmarks): VRAM 99 → 1,412 MiB after `make up-gpu` (1.5B Q4_K_M + KV cache); `--list-devices` shows `CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU`; utilisation ~90% during a 200-token generation. llama.cpp's default log level does not print offload details, so use these checks instead.

| `postgres` | **5433** | `postgres:18.6-alpine` | Tenants, hashed API keys, usage ledger. Data in volume `relay_pgdata` (survives `make down`). Schema: `deploy/compose/initdb/001_schema.sql` |
| `redis` | 6379 | `redis:8.10.2-alpine` | Rate-limit counters. **No persistence** (`--save "" --appendonly no`): counters are short-lived by design |
| `prometheus` | 9090 | `prom/prometheus:v3.15.0` | Scrapes Relay at `host.docker.internal:8000` and llama.cpp every 5 s; evaluates `monitoring/rules.yml`. Data in volume `relay_promdata` (7 days) |
| `grafana` | **3001** | `grafana/grafana:13.2.4` | Dashboard **Relay** provisioned from `monitoring/grafana/` (home page; also `/d/relay-overview`). Anonymous read-only; admin/admin (local only). 3001 because 3000 is often taken |

| `jaeger` | 16686 (UI), 4318 (OTLP/HTTP) | `jaegertracing/jaeger:2.22.0` | Receives Relay's traces when `RELAY_OTEL_ENDPOINT=http://127.0.0.1:4318` (in `.env.example`). In-memory: traces are lost on restart |

### Traces (OpenTelemetry → Jaeger)
- Off unless `RELAY_OTEL_ENDPOINT` is set (tests and CI need no collector). Spans are batched and exported in the background; an unreachable collector drops spans, never slows requests.
- Per request: the server span `POST /v1/chat/completions` (FastAPI instrumentation) with `relay.tenant`, `gen_ai.request.model`, `relay.backend`, `gen_ai.usage.input_tokens` / `output_tokens`, `relay.ttft_ms`, `relay.request_id`, and a `first_token` event; children `relay.auth`, `relay.rate_limit`, `relay.route` → one `relay.backend_attempt` per try (`relay.backend`, `relay.attempt`, `relay.outcome`) → the httpx `POST` to the engine (with a `traceparent` header). Breaker skips are `circuit_open_skip` events.
- No per-chunk spans (`exclude_spans=["send", "receive"]`), no bodies, probes not traced. Every JSON log line written during a request has its `trace_id`: copy it into Jaeger's search (http://localhost:16686) to open the trace.
- Verified 2026-10-13: with the GPU engine stopped, one streamed request's trace showed three ~2 s `BackendUnavailable` attempts on the GPU, then a 0.5 s successful attempt on the CPU, explaining its 6.9 s TTFT.

### Monitoring
- Open **http://localhost:3001** (Grafana) and **http://localhost:9090** (Prometheus: *Status → Targets*, *Alerts*).
- Prometheus expects **Relay on port 8000** (`make run` / `make serve`). If something else holds 8000, the `relay` target shows *down*. (On 2026-10-13 another local project's server held 8000, so the live check used a temporary config pointing at 8010.)
- The dashboard (15 panels): chat requests by status, 5xx ratio (SLO 1 %), backend outcomes; TTFT, inter-token latency and gateway overhead p50/p95/p99 with SLO lines; output tokens/s, tokens/min per tenant, rate-limited requests; breaker state, retries/fallbacks, in-flight (Relay, open streams, llama.cpp processing/queued); limiter degraded, ledger, scrape targets. The JSON in the repo is the source of truth (UI edits are not saved).
- **Alert rules** (`monitoring/rules.yml`, from the brief's SLOs): `RelayHighErrorRate` (>1 % 5xx for 5 m), `RelayTTFTp95AboveSLO` (>1 s for 10 m), `RelayITLp95AboveSLO` (>100 ms for 10 m), `RelayDown`, `RelayCircuitBreakerOpen`, `RelayRateLimiterDegraded`, `RelayLedgerDroppingEvents`. Visible at http://localhost:9090/alerts; no notification delivery (Alertmanager) in the local stack.
- **`make test-alerts`** validates the Prometheus config and runs `monitoring/rules.test.yml` with promtool (e.g. the TTFT alert is pending at 8 min and firing at 20 min when p95 ≈ 1.95 s).
- Verified 2026-10-13: all targets up; 7 rules loaded; Grafana datasource healthy; after mixed traffic and stopping the GPU engine, every panel query returned data (TTFT/ITL percentiles, tokens/s, `relay_circuit_breaker_state{backend="llamacpp-gpu"} = 2`, `up` = 0 for the stopped engine).

**Database helpers:** `make db-shell` (psql), `make redis-cli`, and `make db-reset`, which **deletes all Relay database data** and re-creates the schema.

**Why host port 5433:** this machine also has a native PostgreSQL 18 Windows service (`postgresql-x64-18`) listening on 5432. It is left untouched; Relay's container uses 5433.

**Schema changes need `make db-reset`.** The postgres image runs `initdb/*.sql` **only when its data volume is empty**. Editing `001_schema.sql` later changes nothing until the volume is recreated. This is the trade-off of using plain SQL instead of a migration framework; acceptable here because there is no production data. (PostgreSQL 18 images keep data under `/var/lib/postgresql/18/docker`, so the volume is mounted at `/var/lib/postgresql`, not the older `/var/lib/postgresql/data`.)

**Finding llama.cpp image tags:** the registry lists 12,000+ tags oldest-first in pages of 1000; GitHub release names (`v0.6.0`) differ from image tags (`server-bNNNNN`). Variants include `server-cuda12` and `server-cuda13`.

### OpenAI wire format, as observed from llama.cpp (2026-10-08)

```bash
curl -s localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b-instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":30}'
```

- **Non-streaming:** one `chat.completion` object: `choices[0].message.content`, `finish_reason`, `usage {prompt_tokens, completion_tokens, total_tokens}`. Prompt tokens include the chat template (a 9-word question cost 39 tokens).
- **Streaming** (`"stream": true`): Server-Sent Events. Each event is `data: <json>\n\n` (LF only), ending with `data: [DONE]`.
  1. First chunk: `delta: {"role": "assistant", "content": null}`, with **no text**. TTFT must be measured to the first chunk *with content*.
  2. Content chunks: `delta: {"content": "…"}`.
  3. Final chunk: `delta: {}`, `finish_reason: "stop"`.
  4. With `stream_options: {"include_usage": true}`: an extra chunk with **`choices: []`** and `usage`, before `[DONE]`. Clients that index `choices[0]` on every chunk would break on it; Relay forwards it and reads usage from it.
- **Engine extras:** `timings`, `system_fingerprint`, `prompt_tokens_details.cached_tokens` (prefix cache hits). Relay passes unknown fields through.
- Server logs show `n_threads = 12` on CPU by default.

## 7. Running Relay

```bash
make up      # backends and stores first
make run     # development: Relay on http://localhost:8000, auto-reloads on changes in src/
make serve   # production-style: python -m relay.serve, with graceful shutdown (no reload)
make up-all  # everything in Docker, Relay included (image built from the Dockerfile)
```

### Relay in Docker (`make up-all`)
- `Dockerfile`: `python:3.12-slim` + uv 0.9.9; dependencies installed from `uv.lock` (`--frozen --no-dev`) in a cached layer; runs as non-root user `relay` (uid 10001); `CMD python -m relay.serve`; health check on `/healthz`. `.dockerignore` keeps `.venv`, `.env` and caches out.
- Compose service `relay` (profile `app`) uses `config/relay.docker.yaml` (backends by service name, e.g. `http://llamacpp:8080`) and reaches Postgres, Redis and Jaeger by service name. Host port `RELAY_HOST_PORT` (default 8000). `stop_grace_period: 40s` covers the 5 s grace + 30 s drain.
- Verified 2026-10-13 (on host port 8010): image 438 MB; healthy in ~50 s including the build; contract tests pass against it; `docker stop` during a 120-token stream → `shutdown started: not ready; 1 request(s) in flight`, the stream finished with `[DONE]`, container stopped after ~6 s.

### Liveness, readiness and graceful shutdown
- `GET /healthz` = **liveness** (process alive; stays 200 while draining, so it is not restarted mid-drain). `GET /readyz` = **readiness** (should it get new traffic?): 200 `ready`, or 503 `draining` once shutdown starts; the body also shows `in_flight` and each backend's breaker state.
- With `make serve`, a shutdown signal (SIGTERM; Ctrl+C or Ctrl+Break on Windows) starts **draining**: `/readyz` → 503 and new `/v1` requests → 503 with `Retry-After: 1`, while in-flight requests continue. After `RELAY_SHUTDOWN_GRACE_S` (5 s) uvicorn stops accepting connections; in-flight requests get up to `RELAY_DRAIN_TIMEOUT_S` (30 s) to finish, then are cancelled; then the app shuts down (usage ledger flushed). A second Ctrl+C forces an immediate exit.
- **Windows:** `kill` from Git Bash is `TerminateProcess` (no handler runs, no draining). Use Ctrl+C in the console, or Ctrl+Break sent to a process started with `CREATE_NEW_PROCESS_GROUP` (as `tests/integration/test_graceful_shutdown.py` does). The process exits with code 3 after a Ctrl+Break: uvicorn re-raises the captured signal at the end.
- Verified 2026-10-13: Ctrl+Break during a 150-token stream → `/readyz` 503 (`in_flight: 1`), `/healthz` 200, new request 503; new connections refused after the 2 s grace; the stream still finished with `[DONE]`; the process exited after the app's shutdown.

- **Port 8000**, not 8080: on this machine another project's nginx container already publishes 8080. Docker and uvicorn can both bind it on Windows without an error, and `localhost:8080` then silently reaches nginx. Check with `Get-NetTCPConnection -LocalPort 8000 -State Listen` before blaming Relay.
- Settings come from `RELAY_*` environment variables or `.env` (template: `.env.example`). Today only `RELAY_CONFIG` (default `config/relay.yaml`).
- Endpoints so far: `GET /healthz` (liveness), `GET /v1/models`, `GET /v1/models/{id}` and `POST /v1/chat/completions` (non-streaming and `stream: true`). Invalid request → 400 naming the field (`param`); unknown model → 404 `model_not_found`; backend down → 502; timeout → 504; unknown path / wrong method → 404 / 405; a bug in Relay → 500 `server_error` (traceback in the server log only). A backend's own error keeps its status; its body passes through if OpenAI-shaped and is otherwise wrapped as `backend_error`. **Every** error is in OpenAI format, so the SDK raises the matching class (`BadRequestError`, `NotFoundError`, `RateLimitError`, `InternalServerError`, or `APIError` mid-stream); `tests/unit/test_errors.py` checks each path through the real SDK. Swagger at http://localhost:8000/docs has a pre-filled example body.

### Model routing config (`config/relay.yaml`)
- `backends:` name → `engine` (`llamacpp` | `vllm`), `url`, optional timeouts in seconds: `connect_timeout_s` (5), `first_token_timeout_s` (60), `idle_timeout_s` (30), `total_timeout_s` (300). Each must be ≤ the total.
- `models:` public model name → `backends:` (names, **in order of preference**), optional `owned_by`. Relay currently uses the first backend; fallback down the list comes with routing.
- **Validated at startup; unknown keys are rejected.** A typo stops Relay with a clear error, e.g. `model 'qwen2.5-1.5b-instruct' refers to undefined backends: ['llamacpp-cpuu']` (exit code 1).
- Each backend must serve the model under its **public** name (llama.cpp `--alias`, vLLM `--served-model-name`), so Relay never has to rewrite `model`.

### API keys and tenants
Every `/v1/*` route requires `Authorization: Bearer rk_...` (`/healthz` does not). Swagger's **Authorize** button takes the key.

```bash
uv run relay-admin create-tenant support-bot --name "Support bot" \
    --models qwen2.5-1.5b-instruct --rpm 60 --tpm 20000 --max-tokens 512
uv run relay-admin create-key support-bot --name laptop    # prints the key ONCE
uv run relay-admin list-keys [tenant]                      # prefixes only
uv run relay-admin revoke-key rk_8f3aZ                     # by prefix or key id
```

- **Keys:** `rk_` + 43 URL-safe characters (256 random bits). Postgres stores only SHA-256(key) and an 8-character display prefix. A fast hash is right for random keys; slow hashes (bcrypt/argon2) are for guessable passwords and would cost ~100 ms per request. Verified 2026-10-11: the stored hash equals SHA-256 of the key, and a full table dump contains the raw key 0 times.
- **Cache:** lookups are cached in memory for 30 s (unknown keys 5 s; at most 10,000 entries). **A revoked key keeps working for up to 30 s** (observed: still 200 right after revoking, 401 after 31 s).
- **Errors:** missing key → 401 `missing_api_key`; unknown/revoked key or disabled tenant → 401 `invalid_api_key` (only the prefix is echoed); key store unreachable and key not cached → **503 (fails closed)**. A cached key keeps working while Postgres is down.
- **Startup:** Relay opens its Postgres pool at startup and **refuses to start** if Postgres is unreachable.
- **Tenants created on 2026-10-11 (local DB):** `support-bot` and `batch-summariser` (values from `docs/brief.md`) and `dev` (generous limits, for contract tests). The `dev` key is in `.env` as `RELAY_TEST_API_KEY`; contract tests read it from the environment first, then `.env`. After `make db-reset`, recreate the tenant and key.

### Tenant policy (from the `tenants` row)
- **Model allow-list:** a model must be configured **and** in the tenant's `allowed_models`. A forbidden model gets the **same 404 `model_not_found`** as a nonexistent one (as OpenAI does), so tenants cannot probe which models exist. The reason is logged: `relay.policy tenant X requested model 'Y', which is not in its allow-list (answered 404)`. `/v1/models` lists only the tenant's models.
- **`max_tokens` cap:** `max_tokens`/`max_completion_tokens` above the tenant's cap → 400 naming the field. If the client sends neither, Relay **adds `max_tokens` = cap**, so omitting the field cannot bypass the cap. Verified 2026-10-11: a tenant capped at 5 omitting `max_tokens` got exactly 5 completion tokens (`finish_reason: length`).
- **What Relay changes in a request** (`src/relay/api/upstream.py`): nothing, except adding `stream_options.include_usage` on streams (accounting) and the default `max_tokens` above (policy). When either applies, the JSON is re-serialised once.

### Rate limits (per tenant, token buckets in Redis)
- Two buckets per tenant, sized from its `tenants` row: **requests/min** and **LLM tokens/min**. Each holds up to its per-minute value and refills **continuously** (no burst at minute boundaries).
- **Reserve, then settle:** before a request Relay reserves 1 request + (estimated prompt + `max_tokens`) tokens; afterwards it refunds `reserved - actual` using the engine's count (or charges more if the request used more). A request that fails before generating anything refunds all its tokens. If usage is `missing`, the reservation stands.
- **Atomic:** `src/relay/limits/token_bucket.lua` refills, checks both buckets and takes from both in one Redis script, using Redis's clock (`TIME`). Verified 2026-10-11: 50 concurrent requests against a limit of 10 → **exactly 10 admitted** (3 trials); a naive GET-check-SET version admitted **50**.
- **Keys:** `relay:rl:{tenant}:req` / `:tok` (hash tag `{tenant}` keeps both in one Redis Cluster slot); they expire after the full-refill time + 60 s.
- **Responses:** 429 with `retry-after`, `retry-after-ms` and OpenAI's `x-ratelimit-limit|remaining|reset-requests|tokens` headers (also sent on successful responses). A request that could never fit the token bucket gets 429 "Request too large" without `retry-after`.
- **Inspect:** `make redis-cli`, then `HGETALL "relay:rl:{support-bot}:req"`.
- **Known limitation:** the prompt estimate (chars / 4 + 4 per message) undercounts short prompts, because chat templates add tokens (observed: ~9 reserved vs ~30 real for "hi"). Settling corrects the bucket afterwards, so only the up-front check is looser. Calibrate against measured prompts later rather than guessing a factor now.
- **Redis down ([ADR-002](adr/002-rate-limit-failure-behaviour.md)):** Relay keeps serving and enforces the same limits **per instance** in memory. It logs one warning (`rate limiter: Redis unavailable …`) and one info line on recovery, and skips Redis for 5 s after each failure so requests do not each wait for a timeout. Relay also starts without Redis. Verified by stopping the container: 200, 200, 200, 429 for a 3/min tenant; only the first request paid the ~1 s detection.

### Usage ledger (table `usage_events`)
- One row per request that reached a backend, including failed ones (e.g. 502, with no tokens): request id, time, tenant, key, model, backend, stream, status, prompt/completion tokens, `usage_source`, `ttft_ms` (streams only), `duration_ms`, `cost_gbp`. Every response carries **`x-request-id`**, the row's `request_id`. Requests refused before a backend (401/404/429) are not recorded.
- **Never slows a request:** requests put events on a bounded queue (10,000); a background task writes batches of up to 500 (or every 1 s) in one transaction with `ON CONFLICT (request_id) DO NOTHING`.
- **Postgres down:** the batch is retried with backoff (1 s, 2 s, … up to 30 s) while the queue absorbs new events; each failed attempt logs `relay.ledger usage ledger: write of N events failed …`. **Queue full:** new events are dropped and counted (`dropped`), with a warning at the first drop and every 1,000. **Bad rows** (constraint violations) are retried one by one; only the bad ones are dropped (`rejected`). **Shutdown:** queued and in-flight events are flushed for up to 5 s.
- Verified 2026-10-11 against a running Relay: 5 requests → 5 rows; with Postgres stopped ~10 s, 3 requests still returned 200, three write failures were logged, and all rows appeared after the restart. asyncpg also logs its own `Failed to restore the pool connection floor` tracebacks during an outage; they are expected.
- **`cost_gbp` is NULL for now:** `config/pricing.yaml` rates are `null` (TBD) until the benchmark measures tokens/s; then £/1M = GPU £/h ÷ (tokens/s × 3600) × 1,000,000. No placeholder price is used.
- **Daily report:** `uv run relay-admin usage [tenant] [--days 7]` reads the view **`usage_daily`** (tenant × UTC day × model: requests, failed, prompt/completion/total tokens, `cost_gbp`). The view also counts `requests_without_exact_usage` and `unpriced_requests`, so a sum never looks more complete than it is; the CLI shows "TBD" instead of £0 while rates are unknown, and marks partial sums with `*`.
- **Adding the view to an existing database:** `make db-apply` runs `initdb/002_usage_daily.sql` (`CREATE OR REPLACE VIEW`, safe to repeat) without the data loss of `make db-reset`. Fresh databases get it automatically.

### Routing, fallback and circuit breakers
- Each model's `backends:` list is tried **in order**. A backend that is unreachable, times out or returns 5xx is recorded as a failure and the **next backend** is tried; a 4xx is returned at once (every backend would reject it). The response header **`x-relay-backend`** names the backend that answered; the ledger records it too.
- **Circuit breaker per backend** (`breaker:` in `config/relay.yaml`): after 3 consecutive failures the backend is **open** and skipped without being tried for 10 s; then one **probe** request is let through (half-open); success closes it, failure re-opens it. 4xx responses do not count. Every attempt counts, including retries, and the breaker is checked before each attempt. Transitions are logged: `relay.routing circuit breaker <name>: closed -> open`.
- If every backend is skipped by an open breaker: **503** `service_unavailable`. A mid-stream failure counts against the breaker but never falls back (tokens were already sent).
- Verified 2026-10-13 with GPU llama.cpp first and CPU second: stopping the GPU container → the next request fell back to CPU after ~6 s (3 attempts on Windows), the breaker opened, and following requests went straight to CPU; after restarting the GPU and the 10 s cool-down, one probe closed the breaker and traffic returned to the GPU. All requests returned 200.

### Logs (JSON) and request ids
- Every log line is one JSON object (`timestamp`, `level`, `logger`, `event`, …), including uvicorn's. Relay writes **one `"event": "request"` line per request** when the response is finished: `request_id`, `method`, `path`, `status`, `duration_ms`, and for chat `tenant`, `key` (prefix), `model`, `backend`, `stream`, `prompt_tokens`, `completion_tokens`, `usage_source`, `ttft_ms`. `/healthz`, `/readyz` and `/metrics` are not logged.
- **`request_id`** is assigned in the outermost middleware (before authentication), returned as `x-request-id` on every response, added to every log line written during the request, and used as the usage ledger's `request_id`.
- **Prompts and completions are never logged** (a unit test checks the prompt text is absent).
- Filter on Windows, e.g.: `Get-Content relay.log | ConvertFrom-Json | Where-Object status -ge 500`.

### Metrics (`GET /metrics`, Prometheus text format)
| Metric | Type | Labels |
|---|---|---|
| `relay_http_requests_total`, `relay_http_request_duration_seconds` | counter, histogram | `route` (template), `method`, `status` |
| `relay_ttft_seconds`, `relay_inter_token_latency_seconds` | histogram | `model`, `backend` |
| `relay_gateway_overhead_seconds` | histogram | (Relay's time before calling a backend) |
| `relay_prompt_tokens_total`, `relay_completion_tokens_total`, `relay_cost_gbp_total` | counter | `tenant`, `model` |
| `relay_backend_requests_total` | counter | `backend`, `outcome` (`ok`, `4xx`, `5xx`, `stream_error`) |
| `relay_retries_total`, `relay_fallbacks_total` | counter | `backend` / `model`, `from_backend` |
| `relay_backend_in_flight` (open streams), `relay_in_flight_requests` | gauge | `backend` / – |
| `relay_circuit_breaker_state` (0 closed, 1 half-open, 2 open) | gauge | `backend` |
| `relay_rate_limited_total`, `relay_ratelimiter_degraded` | counter, gauge | `tenant`, `kind` / – |
| `relay_ledger_queue_depth`, `relay_ledger_events_total` | gauge, counter | – / `outcome` |

- **Labels have few fixed values; never ids or text** (a unit test checks no UUID appears in `/metrics`). Latency is in histograms so percentiles can be computed (and aggregated across instances).
- `/metrics` needs no API key, like `/healthz`; in production, expose it only on an internal network.
- `python -m relay.serve` passes `log_config=None` to uvicorn; otherwise uvicorn's own logging config replaces the JSON formatter.

### Request limits (`limits:` in `config/relay.yaml`)
| Limit | Default | Response |
|---|---|---|
| `max_body_bytes` | 1 MB (requirements brief) | 413 |
| `max_messages` | 256 | 400, `param: messages` |
| `max_tokens` (ceiling for `max_tokens` / `max_completion_tokens`) | 4096 | 400, `param` = the field |

- The body limit is an ASGI middleware enforced **while the body is received**. If `Content-Length` is too big, Relay answers 413 before reading anything (curl with `Expect: 100-continue` then uploads 0 bytes). Without `Content-Length` (chunked) it counts bytes and stops once past the limit. A check inside the endpoint would be too late, because FastAPI reads the whole body first.
- The global ceiling applies when the client sends `max_tokens`. If it is omitted, a tenant's own cap is added (see *Tenant policy*); for an uncapped tenant the backend's context window bounds generation (4096 per llama.cpp slot).

### Token usage (accounting input)
- Every completed request produces a `UsageRecord` (model, backend, stream, status, prompt/completion tokens, `source`). Today it is logged as `INFO: relay.usage …`; the usage ledger replaces this sink later.
- **Non-streaming:** read from the response's `usage`.
- **Streaming:** if the client did not set `stream_options.include_usage`, Relay adds it (one of the two documented changes Relay makes to a request, see *Tenant policy*; the JSON is then re-serialised, so whitespace may differ) and removes the engine's usage-only chunk (`choices: []`) before it reaches the client. If the client did set it, the request bytes and the usage chunk pass through unchanged.
- **`source`:** `engine` = the engine's own count (exact). `estimated` = no usage chunk arrived (client disconnected, stream broke, or engine ignores the option): completion ≈ number of content chunks, prompt unknown. `missing` = nothing to go on.
- Verified 2026-10-10 against llama.cpp: a streamed request without `include_usage` recorded the same counts the engine reported for the identical non-streamed request (35 prompt / 24 completion); a disconnect after 5 events recorded `completion=5, source='estimated'`.
- `GET /v1/models` is answered from this file, not by asking backends.

### Running a throwaway Relay in scripts (Windows)
Start `.venv/Scripts/uvicorn.exe` directly, not via `uv run`: killing the `uv run` wrapper from Git Bash can leave the Python child running and holding the port, so a later check silently talks to stale code. Afterwards, confirm the port is free (`Get-NetTCPConnection -LocalPort <port> -State Listen`).

### How streaming behaves (checked 2026-10-10 against llama.cpp)
- Events are forwarded **as soon as each one is complete** (Relay re-cuts TCP chunks into whole `data: …\n\n` events), with `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
- **Before the first byte:** backend unreachable/timeout → normal 502/504 JSON; a backend 4xx/5xx keeps its status and body.
- **Mid-stream failure:** status is already 200, so Relay sends `data: {"error": {...}}` and closes; the OpenAI SDK raises it as an `APIError`. No `[DONE]` follows.
- **Client disconnect cancels the engine's work.** Reproduce: stream a 400-token request, read 5 events, close the connection; llama.cpp logs `stop: cancel task` and releases the slot at ~46 tokens.
- **Timeouts per backend:** *connect* (reach it), *first token* (it starts answering: queueing + prompt processing), *idle* (maximum silence between streamed events), *total* (whole request). A slow but steady stream is fine; a silent one is cut by *idle*.
- **Retries only before the first token** ([ADR-003](adr/003-streaming-retry-policy.md)): `stream()` returns once the backend produced its first event, so everything before that can be retried invisibly. Retried: unreachable backend (incl. connect timeout) and HTTP 503, up to 3 attempts with jittered exponential backoff (`retry:` in `config/relay.yaml`). Not retried: 4xx, other 5xx, first-token/total timeouts. Each retry logs `relay.routing retrying after …`. Observed: against a stopped backend on Windows each refused connection takes ~2 s, so 3 attempts ≈ 6 s before the 502.

## 8. Testing

| Suite | Command | Needs | What it proves |
|---|---|---|---|
| Unit | `make test` | nothing: in-memory fakes | Relay's own logic, including real SDK → Relay → real HTTP backend → fake engine; ~2 s |
| Contract | `make test-contract` | `make up` + `make run` | The **official OpenAI SDK** works against Relay unchanged |
| Integration | `make test-integration` | `make up` (Postgres, Redis; llama.cpp for the shutdown test) | Schema rules, key lookup, ledger, Redis limiter, graceful shutdown under a real signal, and **failure scenarios** (`test_failures.py`). Each DB test runs in a rolled-back transaction, so no data is left behind |

**Failure scenarios** (`tests/integration/test_failures.py`) run scripted engines behind real uvicorn sockets (`tests/engine_server.py`), so failures are genuine (dropped TCP connections, httpx read timeouts):
1. backend crashes after 3 tokens → client gets the 3 tokens and an error event, no `[DONE]`, no retry, breaker counts it;
2. primary too slow to start (first-token timeout) → falls back to the secondary within the timeout; backend silent mid-stream → cut by the idle timeout with `backend_timeout`;
3. Redis unreachable → requests served and still limited per instance (`relay_ratelimiter_degraded 1`);
4. Postgres unreachable after startup → requests served; the ledger buffers events and records write failures.
Stable across repeated runs (~11 s).

- **Test doubles, two levels:** `tests/unit/fakes.py` has `FakeBackend`/`FakeStream` (the `Backend` protocol in memory: gateway logic with no HTTP). `tests/fake_engine.py` has `FakeEngine`, a scripted OpenAI-compatible server used as the httpx transport under the real backend; it mimics llama.cpp's stream (role chunk, token chunks, finish chunk, usage chunk **only when `include_usage` is requested**, `[DONE]`) and records the requests it receives.
- **The OpenAI SDK runs in-process** in unit tests by passing the `TestClient` as `http_client` (`tests/unit/test_errors.py`, `test_end_to_end.py`), so exception classes and parsing are the SDK's own.
- **Check that a test can fail:** e.g. disabling the `include_usage` injection in `chat.py` makes exactly `test_stream_without_include_usage_still_accounts_exactly` fail (checked 2026-10-10).
- Contract tests **skip** locally when Relay is unreachable (the message says what to start). Set `RELAY_REQUIRE_STACK=1` (CI does) to make that a failure instead.
- Overrides: `RELAY_TEST_BASE_URL` (default `http://localhost:8000/v1`), `RELAY_TEST_MODEL` (default `qwen2.5-1.5b-instruct`).
- The SDK client in tests uses `max_retries=0`: by default the OpenAI SDK retries 5xx and timeouts twice, which would hide and multiply failures.
- Markers are applied with `pytestmark` in the test module. A `pytest_collection_modifyitems` hook in a sub-folder `conftest.py` sees **all** collected tests, not just that folder's.

## 9. Benchmarking

```bash
make bench ARGS=--scratch                                      # smoke run (local-smoke.yaml)
make bench SCENARIO=bench/scenarios/<name>.yaml                # real run, saved under results/
make bench-overhead ARGS=--scratch                             # Relay's added latency
```

(`make bench` runs `uv run python -m bench.loadgen <scenario> [--scratch]`.) Needs `make up`
and Relay running (`make run`). The API key is read from the variable named by
`target.api_key_env` (environment or `.env`).

**Load modes.** `mode: open` (the default for results): requests arrive as a Poisson
process at each rate in `arrival_rates_rps`, whether or not earlier ones finished, as real
users do. `mode: closed`: `concurrency` workers each send the next request only when the
previous one finished. A closed loop slows down with the server, so it hides queueing:
use it only to show that effect (`local-smoke-closed.yaml`), never for latency claims.

**Prompt caching.** llama.cpp (and vLLM) reuse cached prompt prefixes across requests. The
workload's seed fixes the lengths and words, but each run and level salts the prompt
openings (`prompt_salt` in `env.json`), so no prompt is ever sent twice. Without the salt,
the 2026-10-16 dry run showed repeated prompts at TTFT 53-102 ms against 585-1107 ms for
new ones on the CPU backend: later levels would have looked much faster than they are.

**Where results go:** `results/run-YYYY-MM-DD-<scenario>-HHMMSS/`, never overwritten:

| File | Contents |
|---|---|
| `config.yaml` | the scenario exactly as run (re-runnable) |
| `results.parquet` | one row per request: timestamps, TTFT/TPOT/E2E, every token time, status, tokens, backend, engine build |
| `summary.json` | per-rate summary (percentiles, throughput, goodput) |
| `env.json` | git SHA + dirty flag, host, GPU + driver (`nvidia-smi`), package versions, engine fingerprints |

`--scratch` writes to `results/_scratch/` (git-ignored): smoke runs and anything run from a
dirty tree are not results. Query any number of runs with DuckDB:

```bash
uv run python -c "import duckdb; print(duckdb.sql(\"select run_id, level, count(*) from 'results/*/results.parquet' group by all\"))"
```

On Windows set `PYTHONIOENCODING=utf-8` first: DuckDB prints box-drawing characters that the
default console codepage cannot encode.

**Cost:** `bench/analysis/cost.py` gives £/1M output tokens = £/h ÷ (tok/s × 3600) × 10⁶,
using `bench/pricing.yaml` (each price with source URL and date; the RTX 4050 laptop has no
public price, so its cost is TBD). These are estimates from assumed list prices, not bills.

**Is the harness accurate?** `tests/integration/test_harness_accuracy.py` runs it against
a fake engine that records when it really sent each token, and checks the harness's own
added time per request (observed on the dev laptop: TTFT +2-6 ms at the median, token gaps
within ~2 ms at p95). The engine's *scripted* delays are not used as the truth: on Windows,
`asyncio.sleep` snaps to the ~15.6 ms system timer, so the engine is slower than its settings.

**Gateway overhead:** `uv run python -m bench.overhead [--scratch]` (needs `make up` for Redis
and Postgres; starts its own Relay on :8010). The same requests go direct to a fake engine
and via Relay; per request it takes *time outside the engine* (harness-observed minus the
engine's recorded timeline), so the engine's jitter cancels out, and compares the two paths.
Relay's server-side `relay_gateway_overhead_seconds` is printed as a cross-check, and the
breakdown is saved as `overhead.json` in the via-Relay run folder. Comparing direct vs via
Relay against llama.cpp instead does not work: the model's own run-to-run variation is far
larger than the overhead.

**Before any timed run:** laptop awake, lid open, on mains power, nothing heavy running. A
test batch on 2026-10-14 failed while Windows entered Modern Standby (lid closed): one run
took 22 minutes and requests stalled. Smoke results like those are discarded, not reported.

### Colab feasibility check (2026-10-10, de-risking only, no measurements)
| Item | Found |
|---|---|
| GPU | Tesla T4, 15,360 MiB, compute capability 7.5; driver 580.82.07 (CUDA 13.0) |
| Python | 3.13.15 |
| vLLM / torch | 0.31.0 / 2.13.0 (`pip install vllm`, ~4 min; resolver warnings concern Colab's preinstalled cudf/cuml/google-adk, not vLLM) |
| Start-up | `/health` 200 after ~180 s, including the model download |

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --served-model-name qwen2.5-1.5b-instruct \
  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8001
```

- **`--dtype half` is required:** the T4 has no bfloat16. (FP8 is also unavailable on this GPU.)
- **Use `--port 8001`:** vLLM defaults to 8000, which is Relay's port; in the notebook both run on one machine. (The check itself ran on 8000 before Relay was involved.)
- Worked: `/v1/models` lists the public name (`owned_by: vllm`); non-streaming completion with `usage`; streaming with `include_usage` ends with a `choices: []` usage chunk, then `data: [DONE]` (same shape as llama.cpp).
- A wrong `model` gets 404 with `"type": "NotFoundError"` and a numeric `"code": 404` (OpenAI uses strings). Relay never forwards unknown models, so clients do not see this.
- vLLM adds many extra response fields (`refusal`, `annotations`, `token_ids`, `kv_transfer_params`, …); Relay passes them through.
- **Conclusion:** the Day 6 plan (vLLM on the free Colab T4) is feasible as written. Not yet checked: AWQ models and the 7B size on the T4 (Day 6, task 6.1).

## 10. Troubleshooting

*(add problems and fixes here as we hit them)*

### `make: command not found` in the VS Code terminal (2026-10-08)
- **Cause:** winget adds `...\WinGet\Packages\ezwinports.make_...\bin` to the **user** PATH, but VS Code and every terminal it spawns keep the PATH from when VS Code started.
- **Fix:** fully quit VS Code (all windows) and reopen it. Reloading one window is not enough.
- **Workaround without restarting** (PowerShell): `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`
- **Verify:** `make --version` prints `GNU Make 4.4.1`.

### `localhost:<port>` answers with someone else's response (404, nginx page) (2026-10-08)
- **Cause:** another Docker container publishes the same port on all addresses, including IPv6 (`[::]:8080`). Relay (uvicorn) listens on IPv4 `127.0.0.1` only. Windows lets both bind, and `localhost` may resolve to `::1`, so requests silently reach the other container. Seen here with `media-archiver-nginx` on 8080 and `nthset-scaffold-1-dashboard-1` on 8001.
- **Check:** `Get-NetTCPConnection -LocalPort 8000 -State Listen` and `docker ps --format "{{.Names}} {{.Ports}}"`.
- **Fix:** keep Relay on a port nothing else publishes (8000 here), or call `127.0.0.1:<port>` explicitly. In scripts, wait for readiness with `curl -f` (fails on 4xx/5xx) so a foreign server answering 404 is not mistaken for Relay.
