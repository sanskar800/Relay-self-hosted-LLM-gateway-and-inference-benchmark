from datetime import date

from relay.cli import format_usage


def row(**overrides) -> dict:
    base = {
        "day": date(2026, 10, 11),
        "tenant_id": "support-bot",
        "model": "qwen2.5-1.5b-instruct",
        "requests": 10,
        "failed_requests": 1,
        "prompt_tokens": 300,
        "completion_tokens": 120,
        "total_tokens": 420,
        "requests_without_exact_usage": 0,
        "cost_gbp": None,
        "unpriced_requests": 10,
    }
    return {**base, **overrides}


def test_unpriced_day_shows_tbd_not_zero() -> None:
    out = format_usage([row()])
    assert "TBD" in out and "0.0000" not in out
    assert "GBP is TBD until benchmark-based rates" in out


def test_partially_priced_day_is_marked() -> None:
    out = format_usage([row(cost_gbp=0.0123, unpriced_requests=3)])
    assert "0.0123*" in out
    assert "3 unpriced" in out


def test_inexact_usage_is_called_out() -> None:
    out = format_usage([row(requests_without_exact_usage=2, cost_gbp=0.5, unpriced_requests=0)])
    assert "2 requests with estimated or missing usage" in out
    assert "lower bound" in out


def test_empty_period() -> None:
    assert "no usage in this period" in format_usage([])
