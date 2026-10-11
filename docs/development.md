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

*(Prometheus, Grafana and Jaeger are added on Day 4.)*

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
make up     # backend first (llama.cpp on :8081)
make run    # Relay on http://localhost:8000, auto-reloads on changes in src/
```

- **Port 8000**, not 8080: on this machine another project's nginx container already publishes 8080. Docker and uvicorn can both bind it on Windows without an error, and `localhost:8080` then silently reaches nginx. Check with `Get-NetTCPConnection -LocalPort 8000 -State Listen` before blaming Relay.
- Settings come from `RELAY_*` environment variables or `.env` (template: `.env.example`). Today only `RELAY_CONFIG` (default `config/relay.yaml`).
- Endpoints so far: `GET /healthz` (liveness), `GET /v1/models`, `GET /v1/models/{id}` and `POST /v1/chat/completions` (non-streaming and `stream: true`). Invalid request → 400 naming the field (`param`); unknown model → 404 `model_not_found`; backend down → 502; timeout → 504; unknown path / wrong method → 404 / 405; a bug in Relay → 500 `server_error` (traceback in the server log only). A backend's own error keeps its status; its body passes through if OpenAI-shaped and is otherwise wrapped as `backend_error`. **Every** error is in OpenAI format, so the SDK raises the matching class (`BadRequestError`, `NotFoundError`, `RateLimitError`, `InternalServerError`, or `APIError` mid-stream); `tests/unit/test_errors.py` checks each path through the real SDK. Swagger at http://localhost:8000/docs has a pre-filled example body.

### Model routing config (`config/relay.yaml`)
- `backends:` name → `engine` (`llamacpp` | `vllm`), `url`, optional `connect_timeout_s` (5), `read_timeout_s` (120).
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
- **Redis down:** not handled yet; decided in ADR-002 (next task).

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
- **Read timeout = maximum silence between chunks**, not total stream duration (`read_timeout_s` per backend in `config/relay.yaml`).

## 8. Testing

| Suite | Command | Needs | What it proves |
|---|---|---|---|
| Unit | `make test` | nothing: in-memory fakes | Relay's own logic, including real SDK → Relay → real HTTP backend → fake engine; ~2 s |
| Contract | `make test-contract` | `make up` + `make run` | The **official OpenAI SDK** works against Relay unchanged |
| Integration | `make test-integration` | `make up` (Postgres, Redis) | Schema rules, key lookup, ledger idempotency, Redis; failure scenarios on Day 4. Each DB test runs in a rolled-back transaction, so no data is left behind |

- **Test doubles, two levels:** `tests/unit/fakes.py` has `FakeBackend`/`FakeStream` (the `Backend` protocol in memory: gateway logic with no HTTP). `tests/fake_engine.py` has `FakeEngine`, a scripted OpenAI-compatible server used as the httpx transport under the real backend; it mimics llama.cpp's stream (role chunk, token chunks, finish chunk, usage chunk **only when `include_usage` is requested**, `[DONE]`) and records the requests it receives.
- **The OpenAI SDK runs in-process** in unit tests by passing the `TestClient` as `http_client` (`tests/unit/test_errors.py`, `test_end_to_end.py`), so exception classes and parsing are the SDK's own.
- **Check that a test can fail:** e.g. disabling the `include_usage` injection in `chat.py` makes exactly `test_stream_without_include_usage_still_accounts_exactly` fail (checked 2026-10-10).
- Contract tests **skip** locally when Relay is unreachable (the message says what to start). Set `RELAY_REQUIRE_STACK=1` (CI does) to make that a failure instead.
- Overrides: `RELAY_TEST_BASE_URL` (default `http://localhost:8000/v1`), `RELAY_TEST_MODEL` (default `qwen2.5-1.5b-instruct`).
- The SDK client in tests uses `max_retries=0`: by default the OpenAI SDK retries 5xx and timeouts twice, which would hide and multiply failures.
- Markers are applied with `pytestmark` in the test module. A `pytest_collection_modifyitems` hook in a sub-folder `conftest.py` sees **all** collected tests, not just that folder's.

## 9. Benchmarking

*(filled in during Days 5–6)*: local runs, the Colab notebook and where results go.

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
