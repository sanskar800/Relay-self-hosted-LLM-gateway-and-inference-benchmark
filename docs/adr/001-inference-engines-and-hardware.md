# ADR-001: Inference engines, hardware and the benchmark matrix

- **Status:** Accepted; amended 2026-10-25 (local runs dropped, see the end)
- **Date:** 2026-10-16

## Context

The original specification benchmarked a 7-8B model at FP16, AWQ and FP8 on a rented
L4/A10 GPU. Relay's rules exclude paid GPUs and APIs (architecture.md §6), so the
hardware is what we have:

| Machine | GPU | VRAM | Notes |
|---|---|---|---|
| Dev laptop | RTX 4050 Laptop (Ada, compute 8.9) | 6 GB (5.75 GiB free at idle) | Windows 11, Docker + NVIDIA runtime |
| Google Colab (free) | Tesla T4 (Turing, compute 7.5) | 15 GB | no bfloat16, no FP8; sessions can end at any time |

Engines: **llama.cpp** (GGUF quantisations, runs on CPU or GPU, already Relay's dev
backend) and **vLLM** (continuous batching and PagedAttention, the common production choice,
Linux + CUDA only, so it runs on Colab). Model family: **Qwen2.5-Instruct** (Apache-2.0;
the 3B is excluded because it uses the Qwen Research licence).

The matrix has to answer questions an engineer would actually ask, using only runs we can
perform, without comparing numbers across different machines.

## Options considered

1. **The specification's matrix (7-8B FP16 / AWQ / FP8):** FP16 7B is about 15 GB of
   weights, leaving no room for a KV cache on a 16 GB T4; the T4 has no FP8. Not feasible.
2. **One engine on one machine:** simple, but cannot show the engine trade-off
   (llama.cpp vs vLLM), which is the most useful finding for choosing a self-hosted stack.
3. **Per-machine matrices, compared only within each machine:** llama.cpp on the laptop
   (CPU vs GPU, quantisation, model size), vLLM on Colab (precision vs AWQ, model size).

## Decision

**Option 3.** Every configuration below gets its own scenario file under `bench/scenarios/`.

**Local RTX 4050, llama.cpp `server-cuda12-b11459`:**

| Config | Model file | Question it answers |
|---|---|---|
| CPU baseline | 1.5B Q4_K_M | the reference without a GPU |
| GPU | 1.5B Q4_K_M | GPU speed-up for the same model and quantisation |
| GPU | 1.5B Q8_0 | cost of higher-precision quantisation (speed vs size) |
| GPU | 7B Q4_K_M | is a 4x larger model usable on a 6 GB laptop GPU? |

7B Q4_K_M is expected to fit just (weights 4.36 GiB + KV cache about 0.22 GiB at a 4096-token
context + about 0.7 GiB of buffers, of 5.75 GiB free). If it does not fit with every layer on
the GPU, it is dropped and recorded as a finding; partial offload would be a different
configuration and is not substituted silently.

**Colab T4, vLLM 0.31.0 (`--dtype half`):** 1.5B FP16, 1.5B AWQ, 7B AWQ. A `max_num_seqs`
sweep and prefix caching on/off are SHOULD items, done only if time allows.

**Engine settings (all llama.cpp configurations):** `--parallel 4 --ctx-size 4096` (4 slots
of 1024 tokens; the workload's largest request is about 900) and `--cache-ram 0` (the
host-RAM prompt cache cannot hit salted prompts and filled the laptop's RAM: findings F9).
Set by `make bench-stack-gpu` / `make bench-stack-cpu`, which also stop every service the
run does not need (F8).

**Method (all configurations):**
- Open-loop Poisson load through Relay, as clients would use it (Relay adds about 4 ms
  before the first token: findings F2). Prompt openings salted per run and level (F1).
- Relay runs with a benchmark config that maps each model to **one** backend with no
  fallback, so a failing backend shows up as errors instead of being hidden behind another
  machine; every result row records the backend that served it, and runs are checked.
- A short probe finds roughly where each configuration saturates; then 4-5 rates from light
  load to past saturation, 120 s each after a 20 s warm-up.
- Same workload (prompt and output length distributions, seed) for every configuration.
- Results compared **only within one machine**; cost per 1M tokens only where a public
  price exists for that GPU (T4: yes; RTX 4050 laptop: TBD).

## Consequences

- Shows CPU vs GPU, quantisation and model size on hardware many developers own, and a
  production-style engine on a datacenter GPU, at £0.
- No cross-engine claims of the form "vLLM is X times faster than llama.cpp": the engines
  ran on different GPUs. The report says so explicitly.
- Colab results depend on which GPU Colab assigns; `env.json` records it, and runs on a
  different GPU are labelled as such, not mixed in.
- Throughput figures are for one small GPU each; they are not capacity planning for
  larger hardware.

## Amendment (2026-10-25): Colab T4 only

**Decision:** the local RTX 4050 matrix is not run; the benchmark is the three Colab
configurations (results committed 2026-10-22/24, findings F10-F11).

**Why:** the laptop proved an unreliable benchmark host. It shares 16 GB of RAM with the
desktop (IDE, browser) and the Docker VM; every timed run there was disturbed: connection
drops (F7), memory exhaustion that stalled every stream and hung Docker (F8), an 8 GiB engine
prompt cache (F9), and its one full run was invalid because a browser grew during it. Each
problem was fixed, but a run still needs an hour of an otherwise idle laptop. Colab gives a
dedicated machine and a GPU with a public price, so it also carries the cost estimates.
(Sustained load is not a hardware risk: the GPU throttles itself when hot.)

**Consequences:**
- No CPU-vs-GPU, Q4_K_M-vs-Q8_0 or llama.cpp benchmark numbers. Quantisation and model size are
  still covered on vLLM (FP16 vs AWQ, 1.5B vs 7B).
- llama.cpp remains Relay's development and CI backend; the engineering findings from the
  local attempts (F7-F9: the keep-alive fix, the memory guard, `--cache-ram 0`) stand.
- The local scenarios, `config/bench/local-*.yaml` and `make bench-stack-*` stay in the repo,
  unrun, so the local matrix can be measured later on a dedicated machine.
