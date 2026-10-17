"""Host memory guard (docs/findings.md F8)."""

import time

import pytest
from bench.loadgen import memory
from bench.loadgen.memory import MemoryMonitor


def test_monitor_samples_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([8.0, 6.0, 0.5, 5.0, 7.0] + [7.0] * 100)
    monkeypatch.setattr(memory, "available_gib", lambda: next(readings))
    with MemoryMonitor(interval_s=0.01) as m:
        time.sleep(0.1)
    report = m.report()
    assert report["available_gib_start"] == 8.0
    assert report["available_gib_min"] == 0.5  # a dip mid-run is caught
    assert report["flagged_low"] is True and m.low


def test_a_healthy_run_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "available_gib", lambda: 4.0)
    with MemoryMonitor(interval_s=0.01) as m:
        time.sleep(0.03)
    assert not m.low and m.report()["available_gib_min"] == 4.0


def test_cli_refuses_to_start_on_a_short_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.loadgen import __main__ as cli

    monkeypatch.setattr(cli, "available_gib", lambda: 0.9)
    with pytest.raises(SystemExit, match="RAM available"):
        cli.main(["bench/scenarios/local-smoke.yaml", "--scratch"])
