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

## F9. llama.cpp's host-RAM prompt cache (8 GiB by default) was the memory hog
*2026-10-17, Day 6 probes on the lean stack (only llama.cpp GPU, Postgres, Redis)*

With the stack trimmed after F8, a probe still drove the host from 2.89 to **0.64 GiB
available** and the new memory guard flagged it invalid
(`results/dev/run-2026-10-17-local-gpu-1.5b-q4-probe-132342`). Inside the Docker VM the
llama.cpp container had grown from 1.23 to 2.66 GiB during the run. Build b11459 keeps a
prompt cache in host RAM, `--cache-ram`, **default 8192 MiB**, and also saves idle slots
into it.

Benchmark prompts are salted (F1), so the cache can never hit: it only costs memory. With
`--cache-ram 0` the same probe kept at least **2.39 GiB available** and the container used
378 MiB (`...-probe-132836`). Results otherwise matched: 0 errors, peak 2.59 req/s.

**Decision:** benchmark runs use `--cache-ram 0` (then set by `make bench-stack-*`, since removed
with the local matrix, and recorded in each scenario's `server_args`); development keeps the engine default (`LLAMACPP_CACHE_RAM`).
For multi-turn chat in production the cache is valuable, but it needs a RAM budget.

**Also:** WSL2 is slow to return freed memory to Windows. Dropping the VM's page cache
(`docker run --rm --privileged alpine sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"`)
shrank the VM from about 4.6 to 2.0 GB within 30 s.

**Open question (from the same probes):** 2 closed-loop workers give barely more
throughput than 1 (0.85 vs 0.78 req/s) and double the inter-token gap (17 vs 8 ms),
while 4 workers give 3.2x. Seen in three probes. One candidate: a new prompt's prefill
sharing a batch with the other request's decoding. To check against the open-loop runs.

## F10. First Colab vLLM runs: AWQ beats FP16 on the T4; the probe was too small for vLLM
*2026-10-22, `bench/colab/relay_benchmark.ipynb` on a free Colab Tesla T4 (driver 580.82.07),
vLLM 0.31.0, code `9b46e84`; runs `run-2026-10-22-colab-t4-*` (the 7B run in `results/`; the
two 1.5B runs, superseded by F11, moved to `results/dev/`), probes and vLLM/Relay logs in
`results/dev/`. All three valid: 0 errors, only `vllm-t4` served, host RAM never
under 7.98 GiB.*

Per-rate numbers come from each run's `summary.json`. Light load (lowest rate) vs the highest
rate tested:

| Config | TPOT p50, light | Highest rate meeting the SLO | Output tok/s there | TTFT p95 there |
|---|---|---|---|---|
| 1.5B FP16 | 16.6 ms | **>= 3.42 req/s** (all 5 rates met it) | 354 | 204 ms |
| 1.5B AWQ | 8.5 ms | **>= 6.05 req/s** (all 5 rates met it) | 639 | 259 ms |
| 7B AWQ | 28.2 ms | **0.84 req/s** (missed from 1.43: ITL p95 137 ms) | 91 | 517 ms |

- **AWQ 4-bit is faster than FP16** for the same 1.5B model: decoding reads every weight from
  GPU memory for each token, and 4-bit weights are a quarter of the bytes. vLLM ran AWQ on the
  **Marlin** kernel (`quantization=auto_awq`, `MarlinLinearKernel`) although `--quantization
  awq` was requested: vLLM 0.31 upgrades it automatically on the T4.
- **Method flaw: the probe stopped at 8 closed-loop workers.** Right for llama.cpp (4 slots),
  wrong for vLLM, which batches far more at once (its log: KV cache room for 171-206 requests
  of 2048 tokens for the 1.5B models). Both 1.5B configs met the SLO at every rate, up to
  1.25x the "peak", so their figures are **lower bounds**, and their costs upper bounds.
  7B AWQ did saturate inside the tested range, so its limit stands. **Fix:** the probe now
  doubles up to 64 workers; the 1.5B configs are re-run with it.
- **`dirty: true` on all three runs is a false positive:** the check counted untracked files,
  and `colab_run.sh` creates its log folder before the harness starts. The code is commit
  `9b46e84`, unchanged on GitHub; the notebook edits nothing and `uv sync --frozen` does not
  modify tracked files. The recorded flag is left as it is; the check now ignores untracked
  files.
- **Logs, checked:** "Cannot use FA version 2" is vLLM noting the T4 (compute 7.5) cannot run
  FlashAttention 2; it used the Triton attention backend. The FP16 log's `EngineDeadError`
  is at 13:12:37, after "[shutdown] API server: shutdown triggered": the script stopping
  vLLM after the run (0 errors in its 1309 requests), not a crash. Prefix cache hit rate
  about 6%: only the chat template's shared system prompt, as for real users (F1).

Estimated cost (T4 at GBP 0.459/h, assumed list price, `bench/pricing.yaml`): 7B AWQ about
GBP 1.41 per 1M output tokens at its limit; the 1.5B figures await the re-run.

## F11. 1.5B re-run with the wider probe: real limits, and AWQ's lead shrinks under load
*2026-10-24, same notebook on a free Colab Tesla T4, vLLM 0.31.0, code `0c4deb2` (`dirty:
false`); runs `results/run-2026-10-24-colab-t4-1.5b-{fp16,awq}-*`, probes and logs in
`results/dev/`. Both valid: 0 errors, only `vllm-t4` served, host RAM never under 7.97 GiB,
scheduling lag p99 at most 14 ms (2 vCPUs kept up).*

The probe now goes to 64 closed-loop workers. Throughput was still rising at 64 (AWQ 7.71 ->
8.39 req/s from 32 to 64 workers, FP16 6.05 -> 7.38), but ITL p95 crossed 100 ms between 32
and 64 workers for both, so this time the open-loop rates bracket the SLO limit:

| Config | Last rate meeting the SLO | First rate missing it | Output tok/s at the limit | TPOT p50, lightest rate |
|---|---|---|---|---|
| 1.5B AWQ | 6.36 req/s (TTFT p95 260 ms, ITL p95 58 ms) | 8.18 req/s (ITL p95 119 ms) | 676 | 9.2 ms |
| 1.5B FP16 | 5.14 req/s (TTFT p95 252 ms, ITL p95 76 ms) | 7.17 req/s (ITL p95 120 ms) | 537 | 18.0 ms |

(The SLO limit lies between the two rates; achieved rates, from `summary.json`.)

- **Reproducible:** where rates overlap, the 22 Oct and 24 Oct sessions (different Colab VMs)
  agree: AWQ ITL p95 55 ms at 6.05 req/s vs 58 ms at 6.36; FP16 47 ms at 3.42 vs 48 ms at 3.79.
- **AWQ's lead shrinks under load:** 2x faster per token alone (9.2 vs 18.0 ms), but only
  1.24x more requests at the SLO limit (6.36 vs 5.14 req/s). A single stream is limited by
  memory bandwidth (reading the weights, where 4-bit helps most); large batches make the GPU
  compute-bound, where AWQ also pays to dequantise its weights.
- **Goodput vs the SLO:** at 8.18 req/s AWQ's goodput was still 100% (each request met TTFT
  <= 1 s and its own average TPOT <= 100 ms), yet the configuration misses the SLO because the
  p95 of all inter-token gaps is 119 ms. The headline uses the brief's p95 SLO, the stricter test.
- **Estimated cost** at the SLO limit (T4 at GBP 0.459/h, assumed list price): 1.5B AWQ
  about GBP 0.19 and 1.5B FP16 about GBP 0.24 per 1M output tokens; 7B AWQ about GBP 1.41 (F10).
- **Logs:** the same harmless FA2 notice and post-run shutdown `EngineDeadError` as F10.

## F12. GSM8K: 7B is far more accurate; AWQ's cost on the 1.5B model is likely but unproven
*2026-10-25, notebook section 4b on a free Colab Tesla T4, vLLM 0.31.0, code `25acfd2`
(`dirty: false`); `results/quality/gsm8k-colab-t4-*`. The same 200 test questions (seed 42),
zero-shot, temperature 0, at most 1024 tokens; 0 errors, 0 truncated, a number found in every
answer, in all three runs.*

| Config | Accuracy | 95% Wilson interval | Mean answer length |
|---|---|---|---|
| 1.5B FP16 | 72.0% (144/200) | 65.4-77.8% | 306 tokens |
| 1.5B AWQ | 65.5% (131/200) | 58.7-71.7% | 288 tokens |
| 7B AWQ | 95.0% (190/200) | 91.0-97.3% | 279 tokens |

Paired comparison (`python -m bench.quality.compare <A> <B>`, exact McNemar test on the
questions where the two disagree, because both answered the same questions):

- **1.5B FP16 vs 1.5B AWQ:** both right 116, both wrong 41, only FP16 right 28, only AWQ right
  15: **p = 0.066**. Suggestive of a real accuracy cost (about 6 points) for AWQ, but not
  conclusive at 200 questions. Wording for the report: "a likely small accuracy cost,
  not proven at this sample size".
- **7B AWQ vs 1.5B AWQ:** only 7B right 62, only 1.5B right 3: **p < 0.0001**. Model size
  matters far more than 4-bit quantisation.

**Trade-off, with F10-F11:** on one T4, 1.5B AWQ serves 6.36 req/s within the SLO at about
GBP 0.19 per 1M output tokens and 65.5% on GSM8K; 7B AWQ serves 0.84 req/s at about GBP 1.41
(7x the cost) and 95.0%. Zero-shot scores compare these configurations with each other only;
they are not comparable with published few-shot GSM8K scores.
