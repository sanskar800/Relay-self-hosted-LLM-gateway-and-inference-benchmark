# Relay

**A self-hosted, OpenAI-compatible LLM gateway and inference benchmark.**

> **Status: under active development (week 1 of build).** Nothing below the "Planned capabilities" heading is finished yet. Every benchmark number in this repository will come from a recorded run in [`results/`](results/). Until then they are marked `TBD`.

---

## What Relay is

Relay sits between your applications and self-hosted open-weight models such as llama.cpp and vLLM. It exposes the standard OpenAI `/v1/chat/completions` API, so any OpenAI SDK client works without code changes. It adds what an organisation needs to share a model fleet safely:

- **Who is calling.** API keys mapped to tenants (teams).
- **What they may do.** Per-tenant model allow-lists, request size limits and rate limits.
- **What it cost.** A per-tenant token and £ usage ledger.
- **Whether it is healthy.** Timeouts, retries, circuit breaking, fallback, metrics and traces.

Relay comes with a **benchmark harness**. The harness measures which serving configuration (engine × quantisation × concurrency) meets a latency target at the lowest cost per million tokens.

> In production you would probably use an existing gateway such as [LiteLLM](https://github.com/BerriAI/litellm) or Envoy AI Gateway. I built Relay to understand the internals of such a gateway and to measure them: streaming correctness, rate-limit semantics, failure behaviour and gateway overhead.

## The problem

UK banks, insurers, healthcare suppliers and public bodies increasingly want open-weight models running inside infrastructure they control, for data residency, cost control and vendor independence. They usually ask two questions first:

1. **Sizing.** Which GPU, engine and quantisation meets our latency target (for example *TTFT p95 < 1 s at N concurrent users*) at the lowest £ per 1M tokens?
2. **Governance.** How do we give 20 internal teams access to the same models with quotas, isolation and cost attribution?

Relay answers both with measurements rather than vendor claims.

## Goals

This project demonstrates:

| Area | Evidence in this repo |
|---|---|
| AI infrastructure | OpenAI-compatible streaming gateway over multiple inference engines |
| Distributed-systems basics | Atomic Redis token bucket, circuit breaker, retries only before the first streamed token, graceful drain |
| Data engineering | Durable usage ledger (PostgreSQL), benchmark results in Parquet, analysis in DuckDB |
| Observability | Prometheus metrics, Grafana dashboard, OpenTelemetry traces, structured logs |
| Performance engineering | Open-loop load generator; TTFT, inter-token latency and p50/p95/p99 percentiles |
| Cost engineering | £ per 1M tokens computed from measured throughput and documented GPU prices |
| Engineering quality | Unit, contract (OpenAI SDK) and integration tests, CI, ADRs |

## Architecture

```text
          Clients (OpenAI SDK, curl, later projects in this portfolio)
                              │  Authorization: Bearer <api key>
                              ▼
┌──────────────────────── Relay gateway (FastAPI, async) ────────────────────────┐
│  auth → tenant → model policy → rate limit (Redis) → routing → reliability     │
│  (timeouts · retries before first token · circuit breaker · fallback)          │
│  SSE streaming passthrough → token accounting → usage ledger (PostgreSQL)      │
└───────────────┬───────────────────────────────┬────────────────────────────────┘
                ▼                               ▼
     llama.cpp server (GGUF)              vLLM (FP16 / AWQ)
     local CPU / local RTX GPU            free Colab GPU (benchmark)

  Metrics → Prometheus → Grafana        Traces → OpenTelemetry → Jaeger
  Benchmark harness → results.parquet → DuckDB → charts + report
```

See [docs/architecture.md](docs/architecture.md) for the full design and why each component exists.

## Technology

Only technologies this project actually uses:

| Concern | Technology |
|---|---|
| Gateway | Python 3.12, FastAPI, httpx (async), Pydantic |
| Inference engines | llama.cpp server (GGUF), vLLM |
| Rate limiting | Redis (Lua token bucket) |
| Usage ledger | PostgreSQL |
| Observability | Prometheus, Grafana, OpenTelemetry, Jaeger, structlog |
| Benchmark & analysis | asyncio load generator, Parquet (PyArrow), DuckDB, Matplotlib |
| Tooling | uv, Docker Compose, pytest, ruff, GitHub Actions, Make |
| Models | Qwen2.5-Instruct family (Apache-2.0): 0.5B for CI, 1.5B for development; benchmark sizes chosen per hardware |

## Current status

| Phase | Status |
|---|---|
| Planning & documentation | ✅ Done |
| Day 1: foundations, local inference, minimal gateway | ✅ Done (GPU backend pending an NVIDIA driver update) |
| Day 2: OpenAI-compatible gateway and streaming | ⏳ Next |
| Day 3: multi-tenancy, rate limiting, usage ledger | ⬜ |
| Day 4: reliability and observability | ⬜ |
| Day 5: benchmark harness | ⬜ |
| Day 6: GPU benchmark (local RTX 4050 + free Colab GPU) | ⬜ |
| Day 7: analysis, report, CI, polish | ⬜ |

The detailed plan is in [TASKS.md](TASKS.md).

## Planned capabilities

- OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming) and `/v1/models`
- API-key authentication with hashed keys and tenant identification
- Per-tenant model allow-lists and request size limits
- Redis token-bucket rate limiting (requests and tokens), with documented behaviour when Redis is down
- Per-tenant token usage and estimated cost in PostgreSQL
- Backend abstraction with routing and fallback across llama.cpp and vLLM
- Per-backend timeouts, retries only before the first token, circuit breaker, graceful shutdown
- Prometheus metrics, Grafana dashboard, OpenTelemetry traces
- Reproducible benchmark harness with open-loop (Poisson) traffic and results saved as Parquet
- Quality check under quantisation (GSM8K subset)

## Benchmark

> **No results yet.** This section will be **generated** from `results/*/results.parquet` by the analysis script. Numbers will not be typed in by hand.

| Metric | Value |
|---|---|
| Gateway overhead p95 | TBD |
| TTFT p95 (at target load) | TBD |
| Inter-token latency p95 | TBD |
| Output throughput (tokens/s) | TBD |
| Max concurrency meeting SLO | TBD |
| Cost / 1M output tokens | TBD |
| GSM8K accuracy (FP16 vs 4-bit) | TBD |

**Hardware used.** Local NVIDIA RTX 4050 Laptop GPU (6 GB) and a free Google Colab GPU. No rented GPUs were used. Cost figures will be *estimates*: measured throughput multiplied by a documented public on-demand GPU price, with the source and date recorded.

## Quick start (local development)

What works today: the OpenAI SDK → Relay → llama.cpp (CPU), **non-streaming**, no API keys yet. The full setup guide is [docs/development.md](docs/development.md).

**Prerequisites:** Docker Desktop, [uv](https://docs.astral.sh/uv/), GNU Make, ~2 GB free disk for models. Ports 8000 (Relay) and 8081 (llama.cpp) must be free.

```bash
git clone https://github.com/sanskar800/Relay-self-hosted-LLM-gateway-and-inference-benchmark.git relay
cd relay
make setup     # Python 3.12 venv + exact dependencies from uv.lock
make models    # pinned Qwen2.5 GGUF models (~1.6 GB) to ~/models, SHA-256 verified
make up        # llama.cpp server on :8081, waits until healthy
make run       # Relay on http://localhost:8000 (leave running; use a second terminal below)
```

Call it with the **official OpenAI SDK**, changing only `base_url` (run with `uv run python`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-for-now")
reply = client.chat.completions.create(
    model="qwen2.5-1.5b-instruct",
    messages=[{"role": "user", "content": "In one sentence, what is an API gateway?"}],
    max_tokens=60,
)
print(reply.choices[0].message.content, reply.usage)
```

Or with `curl` (bash) / PowerShell:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 30}'
```

```powershell
$body = '{"model":"qwen2.5-1.5b-instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'
Invoke-RestMethod http://localhost:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

```bash
make test            # unit tests (no model or Docker needed)
make test-contract   # OpenAI SDK contract tests against the running stack
make down            # stop llama.cpp
```

## Repository layout (target)

```text
src/relay/        gateway: api, auth, limits, routing, backends, accounting, telemetry
bench/            load generator, scenarios (YAML), quality eval, analysis, Colab notebook
results/          committed benchmark runs (config + Parquet + charts + summary)
deploy/compose/   local Docker Compose stack
monitoring/       Prometheus config and rules, Grafana dashboards
tests/            unit / integration / contract
docs/             architecture, development guide, ADRs, report
```

## Documentation

- [docs/brief.md](docs/brief.md): one-page requirements brief (users, tenants, quotas, SLO targets)
- [TASKS.md](TASKS.md): one-week execution plan
- [docs/architecture.md](docs/architecture.md): system design and rationale
- [docs/development.md](docs/development.md): setup guide
- [docs/adr/](docs/adr/): architecture decision records
