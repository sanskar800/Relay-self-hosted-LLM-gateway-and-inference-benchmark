# Development guide

A living setup guide. Each external requirement lists **what it is, why we need it, how to check for it, how to install it and how to verify it.** Update this file whenever setup changes.

## 0. Machine inventory (checked 2026-10-08)

| Item | Found | Notes |
|---|---|---|
| OS | Windows 11 Pro | WSL2 with Ubuntu (Python 3.12.3 inside) |
| CPU | Intel i7-14650HX, 16 cores / 24 threads | Strong CPU baseline for llama.cpp |
| RAM | ~16 GB | Limits how many containers and models run at once |
| GPU | NVIDIA RTX 4050 Laptop, **6 GB** VRAM, compute capability 8.9 (Ada) | ~0.5 GB already used by the desktop |
| NVIDIA driver | 552.27 (CUDA 12.4) | GPU is visible inside WSL2 |
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
- **Python version:** we pin **3.12** with a `.python-version` file. Reason: Google Colab currently runs 3.12, and matching versions avoids "works locally, fails in Colab". uv downloads 3.12 on first `uv sync`; there is nothing to install by hand.

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
  - ~8 GB for the whole Compose stack (models + Postgres + Redis + Grafana + …). If memory gets tight, raise it in `%USERPROFILE%\.wslconfig` (`[wsl2] memory=10GB`).

### make
- **What/why:** short, memorable commands (`make up`, `make test`, `make bench`) that work the same on Linux, macOS, WSL, CI and Colab. All five portfolio repos will share this convention.
- **Check:** `make --version` → GNU Make 4.4.1 ✅ (installed 2026-10-08).
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
| `RELAY_REDIS_URL` | Rate-limit store | `redis://localhost:6379/0` |
| `RELAY_DATABASE_URL` | Usage ledger | `postgresql://relay:relay@localhost:5432/relay` |
| `HF_TOKEN` | Only for gated Hugging Face models (not needed for Qwen2.5) | unset |

## 5. Models

*(filled in during Day 1)*

- Models live **outside** the repo, in `%USERPROFILE%\models` (Windows) or `~/models` (WSL/Colab), mounted read-only into containers.
- Downloaded with a script (`make models`) using `huggingface_hub`, so the exact repository and filename are in version control.
- Candidates (verify licence and availability on Day 1): `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (CI), `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (dev).

## 6. Local stack

*(filled in during Days 1–4)*: `make up`, `make down`, service ports, and how to open Grafana, Prometheus and Jaeger.

## 7. Running Relay

*(filled in during Day 1)*

## 8. Testing

*(filled in during Days 2–4)*: unit, contract (OpenAI SDK) and integration (Compose).

## 9. Benchmarking

*(filled in during Days 5–6)*: local runs, the Colab notebook and where results go.

## 10. Troubleshooting

*(add problems and fixes here as we hit them)*

### `make: command not found` in the VS Code terminal (2026-10-08)
- **Cause:** winget adds `...\WinGet\Packages\ezwinports.make_...\bin` to the **user** PATH, but VS Code and every terminal it spawns keep the PATH from when VS Code started.
- **Fix:** fully quit VS Code (all windows) and reopen it. Reloading one window is not enough.
- **Workaround without restarting** (PowerShell): `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`
- **Verify:** `make --version` prints `GNU Make 4.4.1`.
