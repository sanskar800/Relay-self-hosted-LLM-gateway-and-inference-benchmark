# Findings log

Things learned while building and benchmarking Relay that the final report (Day 7) and
interviews will need. Each entry gives the evidence (a run folder or a test) and how to
reproduce the numbers. **These are development observations, not benchmark results:**
smoke runs at low rates on the CPU backend, recorded to explain decisions. Headline
numbers come only from the Day 6 runs under `results/`.

Evidence runs live in `results/dev/` (committed). Queries over `results/*/results.parquet`
look one level deep, so they never mix these into the benchmark results. On Windows, set
`PYTHONIOENCODING=utf-8` before printing DuckDB tables.

---

## F1. Engines cache prompt prefixes across requests: resent prompts flatter TTFT
*2026-10-16, Day 5 dry run (llama.cpp CPU, Qwen2.5-1.5B Q4_K_M, via Relay)*

The workload was seeded, so every level and every run sent the **same prompts**. llama.cpp
keeps earlier prompts and reuses the longest common prefix (its log: "selected slot by LCP
similarity"). The unique opening on each prompt only protected requests *within* a level.

In the closed-loop run, made right after an open-loop run with the same seed, requests
0-18 had already been sent once:

| Workers | Prompt seen before? | Requests | TTFT p50 |
|---|---|---|---|
| 1 | no | 2 | 1107 ms |
| 1 | yes | 15 | 53 ms |
| 2 | no | 13 | 585 ms |
| 2 | yes | 12 | 102 ms |

```sql
select level, index <= 18 as seen_before, count(*) n, round(median(ttft_s)*1000) ttft_p50_ms
from 'results/dev/run-2026-10-16-local-smoke-closed-180325/results.parquet'
where ok and not in_warmup group by all order by all;
```

The open-loop run was affected the same way: its 0.5 req/s level resent the 0.2 level's
prompts. TTFT p50 at 0.5 req/s was 677 ms before the fix and 944 ms after
(`run-2026-10-16-local-smoke-180159` vs `-180623`).

**Fix:** the seed fixes lengths and words; prompt openings are salted per run and level
(`bench.loadgen.workload.stream`, `prompt_salt` in `env.json`). No prompt is sent twice,
and the workload shape stays identical, so levels and runs remain comparable.

**Takeaway:** a benchmark must never resend a prompt, or it measures the cache, not the
model. Relevant again on Day 6: vLLM's prefix caching is on by default.

## F2. Relay adds about 4 ms before the first token, about 0.1-0.2 ms per stream
*2026-10-14, `python -m bench.overhead` (fixed-latency fake engine, real Relay with Redis
and Postgres, 5 req/s open loop, 118 measured requests per path, dev laptop)*

Per request: time outside the engine = harness-observed time minus the engine's own
recorded timeline, so engine jitter cancels out. Relay adds = via Relay minus direct.

| Relay adds | Run 1 (141149) | Run 2 (141311) |
|---|---|---|
| Before the first token, p50 | 4.05 ms | 3.66 ms |
| Before the first token, p95 | 9.35 ms | 6.15 ms |
| Before the first token, p99 | 32.92 ms | 32.24 ms |
| First to last token, p50 | 0.10 ms | 0.20 ms |
| Server side `relay_gateway_overhead_seconds`, mean | 3.69 ms | 2.77 ms |

Source: `results/dev/run-2026-10-14-overhead-relay-*/overhead.json` (`relay_adds`,
`server_side_mean_s`). "Relay adds" is the difference of the two paths' percentiles, not
a percentile of differences, so tail rows need not be ordered.

- The client-side and server-side figures agree (about 3-4 ms): two methods, one answer.
- **Open question:** the p99 of about 32 ms repeats in both runs, so it is real.
  Candidates: the Postgres ledger's batch flush, Redis round trips under bursts, Python
  garbage collection. Not yet investigated.

## F3. The harness's own measurement error is a few ms
*2026-10-14, `tests/integration/test_harness_accuracy.py` (fake engine over a real socket)*

Against the engine's recorded send times, over repeated runs on the dev laptop: added
TTFT p50 2.4-5.5 ms, token-gap error p95 0.5-2.4 ms. The first request of a run pays about
40-60 ms to open its connection, which the warm-up excludes. Rerun the test to reproduce;
these figures come from its output, not from a results folder.

**Takeaway:** validate the instrument before trusting it, against what actually happened.

## F4. On Windows, scripted sleeps are not ground truth
*2026-10-14*

`asyncio.sleep` snaps to Windows' ~15.6 ms system timer: a scripted 40 ms token gap
measured about 46 ms, and a scripted 20 ms gap about 30 ms. A first attempt at measuring
overhead by comparing two runs' distributions even showed Relay making E2E 33 ms *faster*,
which is impossible. Fixed by comparing against the engine's recorded timeline (F2, F3).

## F5. Closed loops hide overload
*2026-10-16, salted smoke runs (`results/dev/run-2026-10-16-local-smoke-180623`, `-closed-180746`)*

| Mode | Load | TTFT p50 | TTFT p95 | Achieved rate |
|---|---|---|---|---|
| Open | 0.2 req/s offered | 943 ms | 1492 ms | 0.19 req/s |
| Open | 0.5 req/s offered | 944 ms | 2282 ms | 0.43 req/s |
| Closed | 1 worker | 541 ms | 770 ms | 0.43 req/s |
| Closed | 2 workers | 682 ms | 1582 ms | 0.53 req/s |

```sql
select scenario, level, round(median(ttft_s)*1000) p50_ms,
       round(quantile_cont(ttft_s, 0.95)*1000) p95_ms,
       round(count(*) / (max(end_s) - min(sent_s)), 2) achieved_rps
from read_parquet(['results/dev/run-2026-10-16-local-smoke-180623/results.parquet',
                   'results/dev/run-2026-10-16-local-smoke-closed-180746/results.parquet'])
where ok and not in_warmup group by all order by all;
```

At a similar achieved rate (about 0.43 req/s), the closed loop reports TTFT p50 541 ms and
the open loop 944 ms: closed-loop clients never queue, because each waits for its answer.
Too few requests (7-19 per level) for firm conclusions; Day 6 repeats this properly.

## F6. Timed runs need an awake machine
*2026-10-14*

A batch of timing tests failed while Windows entered Modern Standby (lid closed): one run
took 22 minutes and requests stalled. Rule: laptop awake, lid open, on mains power,
nothing heavy running. Results from such runs are discarded.

## F7. llama.cpp's server drops some reused connections; Relay's retries were hiding it
*2026-10-16, Day 6 saturation probe (llama.cpp b11459 CUDA, RTX 4050, 1.5B Q4_K_M)*

The first GPU probe, through a Relay benchmark config with retries off, failed 16-30% of
requests, **even with one worker**: 502 `RemoteProtocolError('Server disconnected without
sending a response.')`, then 503 once the circuit breaker opened
(`results/dev/run-2026-10-16-local-gpu-1.5b-q4-probe-185646`: 212 ok, 67 x 502, 13 x 503).

Isolating it, all with sequential streamed requests of the benchmark's lengths:

| Path | Connections | Failed |
|---|---|---|
| Harness -> llama.cpp directly (no Relay) | pooled | 5 of 44 |
| httpx -> llama.cpp | one reused | 4 of 40 |
| httpx -> llama.cpp | new per request | 0 of 40 |
| Inside the Docker network (no Windows port forwarding) | one reused | 7 of 40 |

Not reproduced with tiny requests (12 back to back) or with idle gaps up to 8 s, so it is
not a simple keep-alive timeout or request cap. Each failure surfaces about 0.21 s after
the request is sent. The engine logged no errors and did not restart.

**Conclusion:** llama.cpp's HTTP server sometimes closes a kept-alive connection after a
streamed response, without telling the client; the next request on it fails. Not Relay,
not Windows, not Docker. Reproduce: `bench/diagnostics/keepalive_reuse.py` (with and
without `--fresh`).

- **Hidden in production config:** ADR-003 retries before the first token caught these, at
  about 0.2 s plus up to 0.1 s backoff of extra TTFT each, on roughly one request in seven.
- **Not seen in F2:** that fake engine runs on uvicorn, which keeps connections correctly.
- **Takeaway:** turn retries off when measuring: they hide failures and add load.

**Side observation (not yet measured properly):** after a few idle seconds the first
request took 290-400 ms instead of about 40 ms, probably the laptop GPU clocking down.
The warm-up period and steady load cover this in benchmark runs.

## F8. On a 16 GB laptop the full stack pages to disk, and timings collapse
*2026-10-16, second Day 6 probe (after the F7 fix; `results/dev/run-2026-10-16-local-gpu-1.5b-q4-probe-192404`)*

With F7 fixed, 1 and 4 closed-loop workers ran without errors (TTFT p95 101 and 127 ms).
Then at 8 workers, about 15 s in, every stream stalled at once for about 15 s; throughput
fell to 0.61 req/s, TTFT p95 reached 14.9 s and two requests hit Relay's 30 s idle timeout
(504). Docker Desktop began answering its own API with HTTP 500 and then stopped responding.

At that moment the laptop had **1.0 GB of 15.7 GB RAM free, 24.9 GB committed and 2,540
pages/s** going to disk. Largest users: the WSL VM running Docker (5.2-6.2 GB: two
llama.cpp engines, Postgres, Redis, Prometheus, Grafana, Jaeger), VS Code (1.6 GB) and
Chrome (1.3 GB).

**Conclusion:** the collapse measured the laptop's memory, not the GPU. The 2-worker level
(no faster than 1 worker) is suspect for the same reason. Probes and runs from this state
are discarded.

**Takeaway:** a benchmark host must run only what the benchmark needs, and the harness
should record memory pressure so a run like this is detected rather than reported.
