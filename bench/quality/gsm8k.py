"""GSM8K accuracy through Relay: does a faster configuration give worse answers?

    uv run python -m bench.quality.gsm8k bench/scenarios/<config>.yaml [--limit N]

Method (identical for every configuration, so only the configuration varies):
- Data: GSM8K test split (grade-school maths word problems, numeric answers), MIT licence,
  pinned by revision and SHA-256. A fixed subset of SUBSET_SIZE questions, chosen with a
  fixed seed, so every configuration answers the same questions.
- Prompt: zero-shot, step by step, ending "The answer is N." Temperature 0, MAX_TOKENS.
- Scoring: exact match of the final number (commas and $ removed) with the reference
  after "####". If the model never writes "The answer is", its last number is used.
- Uncertainty: a 95% Wilson interval. With 200 questions it is several points wide, so
  differences smaller than that are not evidence of a quality difference.

Zero-shot scores are NOT comparable with published few-shot GSM8K scores; they compare our
configurations with each other. Writes results/quality/gsm8k-<scenario>-<time>/.
"""

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from bench.loadgen.__main__ import api_key_for
from bench.loadgen.config import Scenario, load_scenario
from bench.loadgen.results import _git

DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
FILENAME = "main/test-00000-of-00001.parquet"
SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
SUBSET_SIZE = 200
SUBSET_SEED = 42
# A truncated answer scores as wrong, so the cap must not decide the score: at 512 the
# 1.5B model's step-by-step answers were sometimes cut off mid-solution (smoke test,
# 2026-10-25). 1024 + the prompt fits vLLM's --max-model-len 2048; truncations are counted.
MAX_TOKENS = 1024
CONCURRENCY = 16  # throughput only: answers do not depend on it at temperature 0
INSTRUCTION = (
    "Solve the following math problem step by step. "
    "At the end, write the final answer as 'The answer is N.' where N is a number."
)

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
ANSWER_IS = re.compile(r"answer is[:\s]*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def load_questions(cache_dir: Path | None = None) -> list[dict]:
    path = Path(
        hf_hub_download(
            DATASET, FILENAME, repo_type="dataset", revision=REVISION, cache_dir=cache_dir
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{FILENAME}: sha256 {digest}, expected {SHA256}")
    return pq.read_table(path).to_pylist()  # [{"question": ..., "answer": ...}, ...]


def select_subset(n_total: int, size: int = SUBSET_SIZE, seed: int = SUBSET_SEED) -> list[int]:
    """The same question indices for every configuration and every run."""
    return sorted(random.Random(seed).sample(range(n_total), size))


def normalise(number: str) -> Decimal | None:
    try:
        return Decimal(number.replace(",", "").replace("$", "").rstrip("."))
    except InvalidOperation:
        return None


def reference_answer(answer_field: str) -> Decimal:
    """GSM8K puts the final answer after '####' at the end of the worked solution."""
    value = normalise(answer_field.rsplit("####", 1)[1].strip())
    if value is None:
        raise ValueError(f"no reference number in {answer_field[-40:]!r}")
    return value


def extract_answer(text: str) -> Decimal | None:
    """'The answer is N' if present (the last one), otherwise the last number in the text."""
    stated = ANSWER_IS.findall(text)
    if stated:
        return normalise(stated[-1])
    numbers = NUMBER.findall(text)
    return normalise(numbers[-1]) if numbers else None


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def request_body(model: str, question: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }


async def answer_all(
    scenario: Scenario,
    items: list[tuple[int, dict]],
    api_key: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(client: httpx.AsyncClient, index: int, item: dict) -> dict:
        expected = reference_answer(item["answer"])
        row = dict.fromkeys(("extracted", "finish_reason", "completion_tokens", "error", "text"))
        row.update(index=index, expected=str(expected), correct=False)
        async with gate:
            try:
                resp = await client.post(
                    "chat/completions",
                    json=request_body(scenario.target.model, item["question"]),
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                return row
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        got = extract_answer(text)
        row.update(
            text=text,
            extracted=None if got is None else str(got),
            correct=got is not None and got == expected,
            finish_reason=choice.get("finish_reason"),
            completion_tokens=(data.get("usage") or {}).get("completion_tokens"),
        )
        return row

    timeout = httpx.Timeout(scenario.load.request_timeout_s, connect=10)
    async with httpx.AsyncClient(
        base_url=str(scenario.target.base_url), timeout=timeout, transport=transport
    ) as client:
        return await asyncio.gather(*(one(client, i, item) for i, item in items))


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    low, high = wilson_interval(correct, n)
    return {
        "questions": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "ci95": [round(low, 4), round(high, 4)],
        "errors": sum(r["error"] is not None for r in rows),
        "no_number_found": sum(r["error"] is None and r["extracted"] is None for r in rows),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
    }


def write(out_root: Path, scenario: Scenario, rows: list[dict], summary: dict) -> Path:
    now = datetime.now(UTC)
    out = out_root / f"gsm8k-{scenario.name}-{now:%Y-%m-%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows), out / "answers.parquet")
    meta = {
        "summary": summary,
        "scenario": scenario.name,
        "setup": scenario.setup.model_dump() if scenario.setup else None,
        "model": scenario.target.model,
        "dataset": {"repo": DATASET, "revision": REVISION, "file": FILENAME, "sha256": SHA256},
        "method": {
            "subset_size": SUBSET_SIZE,
            "subset_seed": SUBSET_SEED,
            "prompt": INSTRUCTION,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "shots": 0,
        },  # fmt: skip
        "git": _git(),
        "recorded_at": now.isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.quality.gsm8k")
    parser.add_argument("scenario", type=Path, help="benchmark scenario: target, model, setup")
    parser.add_argument("--out", type=Path, default=Path("results/quality"))
    parser.add_argument("--limit", type=int, help="first N of the subset (smoke test only)")
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    questions = load_questions()
    indices = select_subset(len(questions))[: args.limit]
    print(f"-> GSM8K: {len(indices)} questions via {scenario.target.base_url} ({scenario.name})")
    rows = asyncio.run(
        answer_all(scenario, [(i, questions[i]) for i in indices], api_key_for(scenario))
    )
    summary = summarise(rows)
    out = write(args.out / "_scratch" if args.limit else args.out, scenario, rows, summary)
    low, high = summary["ci95"]
    print(
        f"   accuracy {summary['accuracy']:.1%} ({summary['correct']}/{summary['questions']}),"
        f" 95% CI {low:.1%}-{high:.1%}; errors {summary['errors']},"
        f" truncated {summary['truncated']}, no number {summary['no_number_found']}"
    )
    print(f"saved {out}")
    if summary["errors"]:
        raise SystemExit("some requests failed: accuracy is not comparable; see answers.parquet")


if __name__ == "__main__":
    main()
