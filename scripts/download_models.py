"""Download the pinned GGUF models Relay uses, and verify their SHA-256.

Models live outside the repository (default: ~/models, override with RELAY_MODELS_DIR)
and are mounted read-only into the llama.cpp container.

Every model is pinned by repository, filename, revision (Hugging Face commit) and
SHA-256, so a benchmark result always traces back to the exact bytes that produced it.

Usage:
    uv run python scripts/download_models.py          # ci + dev models
    uv run python scripts/download_models.py --all    # also the benchmark-only models
    uv run python scripts/download_models.py --ci     # only the 0.5B CI model
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class PinnedModel:
    purpose: str  # "ci", "dev" or "bench"
    repo_id: str
    filename: str
    revision: str
    sha256: str
    size_bytes: int


# Licence for all entries: Apache-2.0 (Hugging Face model card, checked 2026-10-08).
MODELS: tuple[PinnedModel, ...] = (
    PinnedModel(
        purpose="ci",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        revision="9217f5db79a29953eb74d5343926648285ec7e67",
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        size_bytes=491_400_032,
    ),
    PinnedModel(
        purpose="dev",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
        sha256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        size_bytes=1_117_320_736,
    ),
    PinnedModel(
        purpose="bench",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q8_0.gguf",
        revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
        sha256="d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
        size_bytes=1_894_532_128,
    ),
    # 7B Q4_K_M is published in two parts; llama.cpp is given the first and loads the
    # second from the same folder. Licence checked 2026-10-16: apache-2.0.
    PinnedModel(
        purpose="bench",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        revision="bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        sha256="dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
        size_bytes=3_993_201_344,
    ),
    PinnedModel(
        purpose="bench",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        revision="bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        sha256="539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
        size_bytes=689_872_288,
    ),
)


def models_dir() -> Path:
    return Path(os.environ.get("RELAY_MODELS_DIR", Path.home() / "models")).expanduser()


def sha256_of(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(model: PinnedModel, path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == model.size_bytes
        and sha256_of(path) == model.sha256
    )


def ensure(model: PinnedModel, target_dir: Path) -> Path:
    path = target_dir / model.filename
    if is_valid(model, path):
        print(f"[ok]       {model.filename} (already present, sha256 verified)")
        return path

    print(f"[download] {model.repo_id}/{model.filename} @ {model.revision[:7]}")
    hf_hub_download(
        repo_id=model.repo_id,
        filename=model.filename,
        revision=model.revision,
        local_dir=target_dir,
    )
    if not is_valid(model, path):
        raise SystemExit(f"[error]    {model.filename}: size or sha256 does not match the pin")
    print(f"[ok]       {model.filename} ({model.size_bytes / 1e9:.2f} GB, sha256 verified)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="also download benchmark-only models")
    group.add_argument("--ci", action="store_true", help="only the small CI model")
    args = parser.parse_args(argv)

    purposes = {"ci", "dev", "bench"} if args.all else {"ci"} if args.ci else {"ci", "dev"}
    target_dir = models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {target_dir}")

    for model in MODELS:
        if model.purpose in purposes:
            ensure(model, target_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
