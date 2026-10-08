# Relay: architecture

This is a living document. It describes the **intended** design. Each section notes what is built and what is planned, and it is updated as components land.

## 1. The simplest version

```text
Client ──HTTP──▶ Relay ──HTTP──▶ LLM backend (llama.cpp)
```

At its core Relay is a **reverse proxy that understands one API**: OpenAI chat completions. A client sends `POST /v1/chat/completions` and Relay forwards it to an inference server that speaks the same protocol (llama.cpp and vLLM both do). Relay then passes the answer back, token by token if streaming.

Why put anything in the middle at all? Because the inference server only knows how to *generate tokens*. It does not know:

- who is calling, or whether they are allowed to;
- how much each team has used or what it cost;
- what to do when it is overloaded or crashed;
- how fast it is from the user's point of view.

Each of those is a gateway concern. Putting them in one place means every application gets them for free.

## 2. The full design

```text
                 Clients (OpenAI SDK · curl · later portfolio projects)
                                   │  Authorization: Bearer rk_...
                                   ▼
 ┌─────────────────────────── Relay gateway (FastAPI, asyncio) ───────────────────────────┐
 │                                                                                        │
 │  1. Authentication   hash API key → look up tenant                       (PostgreSQL)  │
 │  2. Tenant policy    model allow-list · max_tokens cap · body size limit               │
 │  3. Rate limiting    token bucket: requests/min + tokens/min per tenant  (Redis + Lua) │
 │  4. Routing          model → ordered backend list; skip unhealthy / open breakers      │
 │  5. Reliability      timeouts · retry only before first token · circuit breaker ·      │
 │                      fallback to next backend · graceful drain on shutdown             │
 │  6. Streaming        SSE passthrough, cancel upstream when the client disconnects      │
 │  7. Accounting       prompt/completion tokens + estimated £ → async usage ledger       │
 │                                                                                        │
 └──────────────┬──────────────────────────────┬──────────────────────────────┬───────────┘
                ▼                              ▼                              ▼
      llama.cpp server               vLLM (OpenAI server)          (optional) secondary
      GGUF Q4/Q8 · CPU or            FP16 / AWQ · GPU               llama.cpp as fallback
      local RTX 4050                 (free Colab GPU)
```

```text
 Observability                              Data / benchmark
 ───────────────                            ─────────────────
 Relay ─ /metrics ─▶ Prometheus ─▶ Grafana  usage events ─▶ PostgreSQL ledger
       ─ OTLP ─────▶ Jaeger (traces)        load generator ─▶ results.parquet ─▶ DuckDB ─▶ charts/report
       ─ stdout ───▶ JSON logs
```

## 3. Components and why each exists

### Gateway: FastAPI + httpx (async)
- **Why:** LLM requests are long-lived (seconds) and I/O-bound. One async process can hold hundreds of open streams without a thread per request.
- **What to watch:** streaming correctness. That means flushing each chunk immediately, propagating client disconnects to the upstream (so we stop paying GPU time for an abandoned answer) and not buffering whole responses in memory.

### Backend abstraction
- **Why:** engines differ in launch flags, health endpoints and how they report token usage. Routing, retries and accounting should not care which engine is behind them.
- **Shape:** a small `Backend` interface (`chat()`, `stream()`, `health()`) with an OpenAI-compatible base class. `llamacpp.py` and `vllm.py` are thin subclasses.

### Authentication and tenants
- **Why:** cost attribution and quotas need an identity. A *tenant* is a team; it can have several API keys.
- **Design:** keys look like `rk_<random>`. We store only a SHA-256 hash, so a database leak does not leak usable keys, and we cache lookups in memory with a short TTL.

### Rate limiting: Redis token bucket
- **Why:** one noisy team must not starve the others of a shared GPU. Limits apply to **requests/min** and **tokens/min**, because a single request can ask for 4,000 tokens.
- **Why Redis:** the bucket state must be shared across gateway replicas. A Lua script makes "refill + check + take" one atomic operation, which avoids race conditions between replicas.
- **Open decision (ADR-002):** what happens when Redis is down? We fail open to a local in-memory limiter for an internal tool, or fail closed for a paid API.

### Routing, reliability and fallback
- **Timeouts:** connect timeout, time-to-first-byte timeout, idle-between-chunks timeout and total timeout, set per backend.
- **Retries:** only **before the first token is sent to the client**. After that, a retry would duplicate or contradict text the user has already seen (ADR-003).
- **Circuit breaker:** after N consecutive failures a backend is marked *open* and skipped for a cool-down period, then probed (*half-open*). This prevents piling requests onto a dead server.
- **Fallback:** the router tries the next backend in the model's list. The spec's paid hosted-API fallback is replaced with a **second local backend**, because this project spends no money.
- **Graceful shutdown:** on SIGTERM, readiness goes false, new requests are refused and open streams drain up to a deadline.

### Usage ledger: PostgreSQL
- **Why:** cost attribution has to survive restarts and be queryable ("tokens and £ per tenant per day"). That is relational data with a durable audit trail.
- **Design:** usage events go into a bounded in-memory queue and a background task writes them in batches. **The ledger never blocks a request.** If Postgres is down, events buffer; if the buffer fills, we drop events and count the drops in a metric.

### Observability
- **Prometheus metrics** (the product of this project): request rate, error rate by backend, TTFT and inter-token latency histograms, output tokens/s, **gateway overhead**, in-flight requests per backend, circuit-breaker state, tokens and £ per tenant.
- **Grafana:** a provisioned dashboard committed as JSON.
- **OpenTelemetry traces:** one span per stage (auth → limit → route → upstream). This shows *where* time goes in one slow request, while metrics show *how often* requests are slow.
- **Logs:** structured JSON with a request ID. Prompt bodies are **not** logged by default, for privacy.

### Benchmark harness
- **Open-loop load generator:** requests arrive on a Poisson schedule whether or not earlier requests have finished, which is how real users behave. A closed-loop generator (N workers, each waiting for its own reply) slows down when the server slows down, so it **hides overload**.
- **Captured per request:** send time, first-token time, per-token arrival times, end time, token counts and status. From these we compute TTFT, inter-token latency (ITL), end-to-end latency, throughput, goodput (requests meeting the SLO) and p50/p95/p99.
- **Gateway overhead:** the same scenario is run *direct to the backend* and *through Relay*. Relay also emits a server-side "time spent in gateway" metric.
- **Storage:** one Parquet row per request, plus the exact `config.yaml` and an `env.json` (GPU, driver, package versions, git SHA). This makes every chart reproducible.
- **Cost:** `£/1M output tokens = GPU £/hour ÷ (measured output tokens/s × 3600) × 1,000,000`. The GPU price is an **input assumption** with a cited source and date, not a measurement.

## 4. Request lifecycle (streaming)

```text
t0  request arrives at Relay
    ├─ auth (cache hit ~µs, miss → Postgres)
    ├─ policy check
    ├─ rate-limit check (1 Redis round-trip)
    ├─ route: pick first healthy backend
t1  upstream request sent              ← gateway overhead (pre)   = t1 - t0
    │   (retry/fallback allowed here)
t2  first token from backend
t3  first token written to client      ← TTFT seen by client      = t3 - t0
    │   (NO retries after this point)
    ├─ stream chunks; record inter-token gaps
t4  final chunk + usage
    └─ enqueue usage event (non-blocking)
```

## 5. Deployment shape

| Environment | What runs where |
|---|---|
| Local development | Redis, PostgreSQL, Prometheus, Grafana, Jaeger and llama.cpp in **Docker Compose**; Relay runs natively with `uv` for fast reload (it can also run in Compose). |
| Local GPU | llama.cpp CUDA container on the RTX 4050 (6 GB). |
| CI | Compose stack with a tiny 0.5B GGUF model on CPU. |
| Final benchmark | Free **Google Colab** GPU. The notebook clones this repo, installs it, starts vLLM and Relay, runs the scenarios, saves results and exits. Colab provides compute only; the repository is the source of truth. |

No permanently running cloud infrastructure. Kubernetes is out of scope (optional extension only).

## 6. Deviations from the original specification

The portfolio roadmap is the source of truth. These changes were **accepted on 2026-10-08** for the constraints of this build (no paid GPU, one week, 6 GB local VRAM):

| Spec | Accepted (2026-10-08) | Reason |
|---|---|---|
| Rented L4/A10 GPU (£20–40) | Local RTX 4050 6 GB + free Colab GPU (typically a T4 with 16 GB) | No spending. |
| 7–8B model, FP16 vs AWQ vs FP8 | Same model *family* (Qwen2.5-Instruct); FP16 vs AWQ on a size that fits the free GPU; 7B AWQ / GGUF Q4 where it fits | FP16 7–8B (~15 GB of weights) does not fit a 16 GB T4 with any KV cache. T4 has no FP8 support. |
| Hosted-API fallback | Second local backend as fallback; hosted fallback optional and off by default | No paid API calls. |
| Measured £/1M tokens on rented GPU | Measured throughput × *documented public on-demand price* for the same GPU class | We cannot measure a bill we never paid; the price is a clearly labelled input. |
| Exact-match cache in gateway | OPTIONAL | Low interview value relative to time. |

Comparisons are only made **within the same hardware**. A T4 number is never presented as if it were measured on an L4.
