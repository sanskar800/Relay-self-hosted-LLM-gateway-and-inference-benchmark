"""Paired comparison of two GSM8K runs on the same questions (exact McNemar test).

    uv run python -m bench.quality.compare results/quality/<run A> results/quality/<run B>

Both runs answer the SAME questions, so comparing their two accuracy intervals throws away
information. McNemar's test looks only at the questions where the two disagree: if A and B
were equally good, "only A right" and "only B right" would each be about half of those.
"""

import argparse
from math import comb
from pathlib import Path

import pyarrow.parquet as pq


def load(run: Path) -> dict[int, bool]:
    return {r["index"]: r["correct"] for r in pq.read_table(run / "answers.parquet").to_pylist()}


def mcnemar(a: dict[int, bool], b: dict[int, bool]) -> dict:
    if a.keys() != b.keys():
        raise ValueError("the runs answered different questions: not a paired comparison")
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    n, k = only_a + only_b, min(only_a, only_b)
    # Exact two-sided binomial test of the disagreements against p = 0.5.
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {
        "questions": len(a),
        "both_right": sum(a[i] and b[i] for i in a),
        "both_wrong": sum(not a[i] and not b[i] for i in a),
        "only_a_right": only_a,
        "only_b_right": only_b,
        "p_value": p,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.quality.compare")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)
    r = mcnemar(load(args.run_a), load(args.run_b))
    a, b = args.run_a.name, args.run_b.name
    print(f"A = {a}\nB = {b}")
    print(
        f"{r['questions']} questions: both right {r['both_right']}, both wrong {r['both_wrong']}, "
        f"only A right {r['only_a_right']}, only B right {r['only_b_right']}"
    )
    print(f"exact McNemar p = {r['p_value']:.4f}")


if __name__ == "__main__":
    main()
