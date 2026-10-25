"""GSM8K scoring rules: what counts as a correct answer, and how sure we can be."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import yaml
from bench.loadgen.config import Scenario
from bench.quality.gsm8k import (
    answer_all,
    extract_answer,
    reference_answer,
    select_subset,
    summarise,
    wilson_interval,
)


def test_reference_is_the_number_after_the_hashes() -> None:
    assert reference_answer("Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18") == Decimal(18)
    assert reference_answer("...\n#### 1,250") == Decimal(1250)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3 + 4 = 7. The answer is 7.", Decimal(7)),
        ("The answer is $1,250.", Decimal(1250)),
        ("First 12, then 5. The answer is: 18", Decimal(18)),
        ("I think it is 4, no, the answer is 6. Wait, the answer is 8.", Decimal(8)),  # last
        ("So she has 42 apples left.", Decimal(42)),  # no phrase: last number
        ("The answer is 2.50", Decimal("2.5")),  # 2.50 == 2.5
        ("I cannot solve this.", None),
    ],
)
def test_extract_answer(text: str, expected: Decimal | None) -> None:
    assert extract_answer(text) == expected


def test_subset_is_fixed_and_the_same_for_everyone() -> None:
    a, b = select_subset(1319), select_subset(1319)
    assert a == b and len(a) == 200 and len(set(a)) == 200
    assert a == sorted(a) and max(a) < 1319


def test_wilson_interval_is_honest_about_200_questions() -> None:
    low, high = wilson_interval(120, 200)  # 60%
    assert low == pytest.approx(0.531, abs=0.002) and high == pytest.approx(0.665, abs=0.002)
    assert high - low > 0.13  # about +-7 points: small differences are noise
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_answers_are_scored_through_the_api() -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["messages"][0]["content"]
        text = "The answer is 18." if "eggs" in question else "So 5."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 9},
            },
        )

    raw = yaml.safe_load(Path("bench/scenarios/local-smoke.yaml").read_text(encoding="utf-8"))
    raw["target"] = {"base_url": "http://relay/v1", "model": "m"}
    scenario = Scenario.model_validate(raw)
    items = [
        (0, {"question": "How many eggs?", "answer": "#### 18"}),
        (1, {"question": "How many cats?", "answer": "#### 4"}),
    ]
    rows = asyncio.run(answer_all(scenario, items, None, httpx.MockTransport(reply)))
    assert [r["correct"] for r in rows] == [True, False]
    s = summarise(rows)
    assert (s["questions"], s["correct"], s["accuracy"], s["errors"]) == (2, 1, 0.5, 0)


def test_mcnemar_uses_only_the_disagreements() -> None:
    from bench.quality.compare import mcnemar

    # 10 questions: 6 both right, 1 both wrong, 3 only A right, 0 only B right.
    a = {i: i < 9 for i in range(10)}
    b = {i: i < 6 for i in range(10)}
    r = mcnemar(a, b)
    assert (r["both_right"], r["both_wrong"], r["only_a_right"], r["only_b_right"]) == (6, 1, 3, 0)
    assert r["p_value"] == pytest.approx(0.25)  # 2 * (1/2)^3: three disagreements prove little
    assert mcnemar(a, a)["p_value"] == 1.0
    with pytest.raises(ValueError, match="different questions"):
        mcnemar(a, {0: True})
