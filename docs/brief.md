# Relay: requirements brief

*One page. Every feature in [TASKS.md](../TASKS.md) should trace back to a line here. Written 2026-10-08.*

## Problem

An organisation runs open-weight LLMs on its own GPUs and wants several internal teams to share them. It needs to know **who is using the models, within what limits, at what cost**, and **which serving configuration meets its latency target most cheaply**.

## Users

| User | Needs |
|---|---|
| **Application developer** (in a tenant team) | Call models with the standard OpenAI SDK by changing only `base_url` and the API key. Streaming responses. Clear errors (401/403/429/5xx) in OpenAI format. |
| **Platform operator** | Create tenants and keys, set limits, see health, latency and per-tenant usage and £ on a dashboard. Survive a backend failure without paging anyone. |
| **Capacity planner** | A reproducible benchmark that answers "which engine × quantisation × concurrency meets our SLO at the lowest £ per 1M tokens?" |

## Tenants and quotas (demo configuration)

A **tenant** is a team. It owns one or more API keys, and limits apply per tenant, not per key.

| Tenant | Workload | Allowed models | Requests/min | Tokens/min | `max_tokens` cap |
|---|---|---|---|---|---|
| `support-bot` | Interactive chat, latency-sensitive | `qwen2.5-1.5b-instruct` | 60 | 20,000 | 512 |
| `batch-summariser` | Offline summarisation, throughput-oriented | `qwen2.5-1.5b-instruct`, `qwen2.5-0.5b-instruct` | 30 | 60,000 | 2,048 |

These values are **configuration defaults** for demos and tests, not capacity claims. Request body limit: 1 MB for every tenant.

## Service level objectives (**targets, not results**)

| SLO | Target | Measured on |
|---|---|---|
| Time to first token (TTFT) p95 | **< 1 s** | Interactive scenario, through Relay |
| Inter-token latency (ITL) p95 | **< 100 ms** (≈ 10 tokens/s, faster than reading speed) | Same |
| Gateway overhead p95 | **< 10 ms** (spec: "a few milliseconds") | Relay vs direct-to-backend, same scenario |
| Error rate (excluding intended 429s) | **< 1 %** | Same |

The benchmark's headline output is **the highest load (requests/s, concurrent streams) at which each configuration still meets the TTFT and ITL targets**, plus its £/1M output tokens. Until measured, all of these are `TBD`.

## Functional requirements

1. OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`.
2. API keys (`rk_…`) mapped to tenants; only hashes stored.
3. Per-tenant model allow-list, `max_tokens` cap and body size limit.
4. Per-tenant rate limits on requests/min **and** tokens/min, shared across gateway replicas.
5. Per-request usage record: tenant, model, backend, tokens, latency, estimated £.
6. At least two backends behind one interface (llama.cpp, vLLM), with fallback when one fails.

## Non-functional requirements

- **Reliability:** timeouts, retries only before the first streamed token, circuit breaker, graceful drain on shutdown. A Redis or Postgres outage must not take requests down (behaviour documented in ADRs).
- **Privacy:** prompts and completions are not logged by default.
- **Observability:** metrics for every SLO above, per-tenant usage, traces per request stage.
- **Reproducibility:** one-command local stack; every benchmark number regenerable from committed Parquet.
- **Cost:** £0 to build and run (local RTX 4050, free Colab GPU).

## Out of scope

SSO/OAuth and user accounts · billing and invoicing · content moderation and guardrails · multi-region and Kubernetes · fine-tuning · a web UI beyond Grafana · paid hosted-API fallback (optional, off by default) · semantic caching.
