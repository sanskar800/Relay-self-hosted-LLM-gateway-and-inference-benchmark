"""Generate every benchmark table, chart and report from the result files (tasks 7.1-7.3).

    uv run python -m bench.analysis.analyze        # or: make report

Reads   results/run-*/          (one benchmark run per configuration: Parquet + metadata)
        results/quality/gsm8k-*/  (GSM8K answers)
        results/dev/            (the probes that chose each run's rates; overhead evidence)
        bench/pricing.yaml      (assumed GPU prices)
Writes  results/run-*/summary.md, results/charts/*.png, docs/report.md

Nothing is typed by hand: delete the outputs and rerun to get the same files. Every
metric is recomputed in SQL (queries.sql) from the raw per-request timestamps.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from bench.analysis.cost import LABEL as COST_LABEL  # noqa: E402
from bench.analysis.cost import cost_per_million_tokens, load_pricing  # noqa: E402
from bench.quality.compare import mcnemar  # noqa: E402
from bench.quality.gsm8k import wilson_interval  # noqa: E402

RESULTS = Path("results")
CHARTS = RESULTS / "charts"
REPORT = Path("docs/report.md")
QUERIES = Path(__file__).with_name("queries.sql")

# Brief SLO (docs/brief.md): TTFT p95 < 1 s, ITL p95 < 100 ms, error rate < 1%.
SLO_TTFT_P95_S = 1.0
SLO_ITL_P95_S = 0.1
SLO_MAX_ERROR_RATE = 0.01

# Categorical slots 1-3 of the reference palette, validated all-pairs in light mode
# (dataviz validate_palette.js); slot 3 is below 3:1 contrast, so every series is also
# labelled directly and every chart has a table in the report.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


@dataclass(frozen=True)
class Run:
    run_id: str
    scenario: str
    config: dict
    env: dict

    @property
    def label(self) -> str:
        s = self.config["setup"]
        size = "7B" if "7b" in self.scenario else "1.5B"
        return f"{size} {s['quantization']}"

    @property
    def gpu_class(self) -> str | None:
        name = " ".join(g["name"] for g in self.env.get("gpu") or [])
        return "t4" if "T4" in name else ("rtx-4050-laptop" if "4050" in name else None)


def load_queries() -> dict[str, str]:
    sql = QUERIES.read_text(encoding="utf-8")
    pattern = r"-- name: (\w+)\n(.*?)(?=\n-- name:|\Z)"
    return {m.group(1): m.group(2).strip() for m in re.finditer(pattern, sql, re.S)}


def load_runs() -> list[Run]:
    runs = []
    for d in sorted(RESULTS.glob("run-*")):
        runs.append(
            Run(
                run_id=d.name,
                scenario=yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["name"],
                config=yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")),
                env=json.loads((d / "env.json").read_text(encoding="utf-8")),
            )
        )
    # A stable, meaningful order: by model size, then precision.
    return sorted(runs, key=lambda r: ("7b" in r.scenario, r.config["setup"]["quantization"]))


def connect(runs: list[Run]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    files = [str(RESULTS / r.run_id / "results.parquet") for r in runs]
    con.execute(f"CREATE VIEW requests AS SELECT * FROM read_parquet({files})")
    con.execute(
        "CREATE TABLE runs (run_id VARCHAR, scenario VARCHAR, warmup_s DOUBLE,"
        " slo_ttft_s DOUBLE, slo_tpot_s DOUBLE)"
    )
    con.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        [
            (r.run_id, r.scenario, r.config["load"]["warmup_s"],
             r.config["slo"]["ttft_s"], r.config["slo"]["tpot_s"])
            for r in runs
        ],
    )  # fmt: skip
    answers = sorted((RESULTS / "quality").glob("gsm8k-*/answers.parquet"))
    if answers:
        con.execute(
            "CREATE VIEW answers AS SELECT regexp_extract(filename, '(gsm8k-[^/\\\\]+)[/\\\\]', 1)"
            f" AS run_id, * FROM read_parquet({[str(a) for a in answers]}, filename = true)"
        )
    return con


def levels(con, queries) -> list[dict]:
    return [dict(zip(c, row, strict=True)) for c, row in _rows(con, queries["levels"])]


def _rows(con, sql: str):
    rel = con.sql(sql)
    cols = rel.columns
    return ((cols, row) for row in rel.fetchall())


def meets_slo(level: dict) -> bool:
    return (
        level["ttft_p95"] is not None
        and level["ttft_p95"] < SLO_TTFT_P95_S
        and level["itl_p95"] is not None
        and level["itl_p95"] < SLO_ITL_P95_S
        and level["error_rate"] < SLO_MAX_ERROR_RATE
    )


def slo_limit(run_levels: list[dict]) -> tuple[dict | None, dict | None]:
    """(last level meeting the SLO, first level missing it), in order of offered load.
    The capacity lies between them; if nothing missed, it is only a lower bound."""
    last_ok = None
    for level in sorted(run_levels, key=lambda x: x["level"]):
        if not meets_slo(level):
            return last_ok, level
        last_ok = level
    return last_ok, None


# --- charts -------------------------------------------------------------------------------


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def _save(fig, name: str) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / name
    fig.tight_layout()
    # No Software/date metadata: regenerating gives byte-identical files.
    fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


def _plain_ms(axis) -> None:
    """Log-scale ticks as plain numbers (100, 200, 500) instead of 10^2 notation."""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else "")
    )


def _threshold(ax, y: float, text: str) -> None:
    ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color=MUTED, xytext=(0, 3), textcoords="offset points")  # fmt: skip


def _label_end(ax, x: float, y: float, text: str) -> None:
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK)  # fmt: skip


def latency_chart(runs, by_run, metric: str, slo: float, title: str, ylabel: str, name: str):
    # Offered load on x: it is what the test controls, and it keeps rising past saturation
    # (served load flattens there, which would fold the line back on itself).
    fig, ax = _axes(title, "Offered load (requests per second, open loop)", ylabel)
    for i, run in enumerate(runs):
        pts = sorted(by_run[run.run_id], key=lambda x: x["level"])
        xs = [p["level"] for p in pts]
        ys = [p[metric] * 1000 for p in pts]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=run.label)  # fmt: skip
        _label_end(ax, xs[-1], ys[-1], run.label)
    ax.set_yscale("log")
    _plain_ms(ax.yaxis)
    _threshold(ax, slo * 1000, f"SLO {slo * 1000:.0f} ms")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    ax.margins(x=0.12)
    return _save(fig, name)


def probe_chart(runs, probes: dict[str, list[dict]]):
    fig, ax = _axes(
        "Throughput vs concurrent users (closed-loop probe)",
        "Concurrent users (closed loop)",
        "Output tokens per second",
    )
    for i, run in enumerate(runs):
        pts = probes.get(run.run_id) or []
        if not pts:
            continue
        xs = [p["level"] for p in pts]
        ys = [p["output_tokens_per_s"] for p in pts]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=run.label)  # fmt: skip
        _label_end(ax, xs[-1], ys[-1], run.label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.margins(x=0.15)
    return _save(fig, "throughput_vs_concurrency.png")


def itl_distribution_chart(runs, gaps: dict[str, list[float]], at: dict[str, float]):
    fig, ax = _axes(
        "Inter-token latency at each configuration's SLO limit",
        "Gap between tokens (ms, log scale)",
        "Share of gaps at or below",
    )
    for i, run in enumerate(runs):
        g = sorted(x * 1000 for x in gaps.get(run.run_id, []) if x > 0)
        if not g:
            continue
        ys = [(k + 1) / len(g) for k in range(len(g))]
        label = f"{run.label} @ {at[run.run_id]:.2f} req/s"
        ax.plot(g, ys, color=SERIES[i], linewidth=2, label=label)
    ax.set_xscale("log")
    # Near-zero gaps (tokens arriving in one network read) would stretch the axis.
    ax.set_xlim(3, 1000)
    _plain_ms(ax.xaxis)
    ax.axvline(SLO_ITL_P95_S * 1000, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.annotate("SLO: p95 under 100 ms", xy=(SLO_ITL_P95_S * 1000, 0.5), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=MUTED)  # fmt: skip
    ax.annotate("p95", xy=(3, 0.95), xytext=(3, -10), textcoords="offset points", fontsize=8,
                color=MUTED)  # fmt: skip
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=8, loc="center left", labelcolor=INK)
    return _save(fig, "itl_distribution.png")


def cost_chart(rows: list[dict]):
    fig, ax = _axes(
        "Estimated cost per 1M output tokens at the SLO limit (T4)",
        "",
        "GBP per 1M output tokens (estimate)",
    )
    names = [r["label"] for r in rows]
    values = [r["cost"] for r in rows]
    bars = ax.bar(names, values, color=SERIES[: len(rows)], width=0.55, edgecolor=SURFACE,
                  linewidth=2)  # fmt: skip
    for bar, r in zip(bars, rows, strict=True):
        text = f"GBP {r['cost']:.2f}"
        if r.get("accuracy") is not None:
            text += f"\nGSM8K {r['accuracy']:.1%}"
        ax.annotate(text, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)  # fmt: skip
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis="x", visible=False)
    return _save(fig, "cost_per_million_tokens.png")


# --- tables --------------------------------------------------------------------------------


def ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.0f} ms"


def level_table(rows: list[dict]) -> str:
    head = (
        "| Offered (req/s) | Served (req/s) | Errors | TTFT p50 | TTFT p95 | ITL p95 "
        "| TPOT p50 | Output tok/s | Goodput | SLO |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['level']:.2f} | {r['throughput_rps']:.2f} | {r['errors']} | {ms(r['ttft_p50'])} "
        f"| {ms(r['ttft_p95'])} | {ms(r['itl_p95'])} | {ms(r['tpot_p50'])} "
        f"| {r['output_tokens_per_s']:.0f} | {r['goodput_ratio']:.0%} "
        f"| {'meets' if meets_slo(r) else '**misses**'} |\n"
        for r in sorted(rows, key=lambda x: x["level"])
    )
    return head + body


def run_summary(run: Run, rows: list[dict]) -> str:
    s = run.config["setup"]
    return (
        f"# {run.scenario}\n\n"
        "Generated by `python -m bench.analysis.analyze` from this folder's `results.parquet`.\n\n"
        f"- **Hardware:** {s['hardware']}\n- **Engine:** {s['engine']} `{s['server_args']}`\n"
        f"- **Model:** {s['model_file']} ({s['quantization']})\n"
        f"- **Code:** {run.env['git']['sha']}\n\n" + level_table(rows)
    )


# --- report --------------------------------------------------------------------------------


def generate() -> Path:
    queries = load_queries()
    runs = load_runs()
    if not runs:
        raise SystemExit("no results/run-* folders")
    con = connect(runs)
    all_levels = levels(con, queries)
    by_run: dict[str, list[dict]] = {r.run_id: [] for r in runs}
    for row in all_levels:
        by_run[row["run_id"]].append(row)

    for run in runs:
        (RESULTS / run.run_id / "summary.md").write_text(
            run_summary(run, by_run[run.run_id]), encoding="utf-8", newline="\n"
        )

    # Probes that chose each run's rates (closed loop, results/dev).
    probes: dict[str, list[dict]] = {}
    for run in runs:
        probe = (run.env.get("rates_from_probe") or {}).get("probe_run")
        if probe and (RESULTS / "dev" / probe / "summary.json").exists():
            probes[run.run_id] = json.loads(
                (RESULTS / "dev" / probe / "summary.json").read_text(encoding="utf-8")
            )

    # SLO limits, costs, quality.
    pricing = load_pricing()
    limits = {r.run_id: slo_limit(by_run[r.run_id]) for r in runs}
    quality = {}
    if "answers" in [t[0] for t in con.sql("SHOW TABLES").fetchall()]:
        for cols, row in _rows(con, queries["quality"]):
            q = dict(zip(cols, row, strict=True))
            quality[q["run_id"]] = q
    quality_by_scenario = {
        s: q for s in (r.scenario for r in runs) for k, q in quality.items() if f"gsm8k-{s}-" in k
    }

    headline = []
    for run in runs:
        last_ok, first_miss = limits[run.run_id]
        price = pricing.gbp_per_hour(run.gpu_class) if run.gpu_class else None
        cost = cost_per_million_tokens(price, last_ok["output_tokens_per_s"]) if last_ok else None
        q = quality_by_scenario.get(run.scenario)
        headline.append(
            {"run": run, "label": run.label, "ok": last_ok, "miss": first_miss, "cost": cost,
             "accuracy": q["accuracy"] if q else None, "quality": q}
        )  # fmt: skip

    # Charts.
    latency_chart(runs, by_run, "ttft_p95", SLO_TTFT_P95_S, "Time to first token, p95, vs load",
                  "TTFT p95 (ms, log scale)", "ttft_p95_vs_load.png")  # fmt: skip
    latency_chart(runs, by_run, "itl_p95", SLO_ITL_P95_S, "Inter-token latency, p95, vs load",
                  "ITL p95 (ms, log scale)", "itl_p95_vs_load.png")  # fmt: skip
    probe_chart(runs, probes)
    gaps: dict[str, list[float]] = {r.run_id: [] for r in runs}
    at = {}
    for h in headline:
        if h["ok"]:
            at[h["run"].run_id] = h["ok"]["throughput_rps"]
    for cols, row in _rows(con, queries["itl_samples"]):
        g = dict(zip(cols, row, strict=True))
        limit = limits[g["run_id"]][0]
        if limit and g["level"] == limit["level"]:
            gaps[g["run_id"]].append(g["gap"])
    itl_distribution_chart(runs, gaps, at)
    cost_chart([h for h in headline if h["cost"] is not None])

    REPORT.write_text(render_report(runs, by_run, headline, probes), encoding="utf-8", newline="\n")
    return REPORT


def render_report(runs, by_run, headline, probes) -> str:
    pricing = load_pricing()
    t4 = pricing.gpus["t4"]
    lines = [
        "# Relay benchmark report",
        "",
        "> **Generated** by `python -m bench.analysis.analyze` (`make report`) from the files in "
        "[`results/`](../results/). Do not edit by hand: change the data or the generator and "
        "regenerate. Every number below is recomputed in DuckDB from the raw per-request "
        "timestamps (`bench/analysis/queries.sql`).",
        "",
        "## Headline: the highest load each configuration serves within the SLO",
        "",
        f"SLO (docs/brief.md): **TTFT p95 < {SLO_TTFT_P95_S:.0f} s, ITL p95 < "
        f"{SLO_ITL_P95_S * 1000:.0f} ms, error rate < {SLO_MAX_ERROR_RATE:.0%}**, through Relay, "
        "open-loop Poisson arrivals. The true limit lies between the last rate that met the SLO "
        "and the first that missed it.",
        "",
        "| Configuration | Meets SLO up to | Misses from | Output tok/s at the limit "
        "| TTFT p95 / ITL p95 there | Est. GBP per 1M output tokens | GSM8K (95% CI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in headline:
        ok, miss, q = h["ok"], h["miss"], h["quality"]
        acc = "-"
        if q:
            low, high = wilson_interval(q["correct"], q["questions"])
            acc = f"{q['accuracy']:.1%} ({low:.1%}-{high:.1%})"
        lines.append(
            f"| {h['label']} | {ok['throughput_rps']:.2f} req/s | "
            + (f"{miss['throughput_rps']:.2f} req/s" if miss else "not reached (lower bound)")
            + f" | {ok['output_tokens_per_s']:.0f} | {ms(ok['ttft_p95'])} / {ms(ok['itl_p95'])} | "
            + (f"{h['cost']:.2f}" if h["cost"] is not None else "TBD")
            + f" | {acc} |"
        )
    lines += [
        "",
        f"Cost is an **{COST_LABEL}**: a T4 at USD {t4.usd_per_hour}/h x "
        f"{pricing.fx.usd_to_gbp} GBP/USD (AWS g4dn.xlarge on-demand, eu-west-2; source and date in "
        "`bench/pricing.yaml`), divided by the measured output tokens per second, assuming the GPU "
        "is busy all hour.",
        "",
        "![TTFT p95 vs load](../results/charts/ttft_p95_vs_load.png)",
        "",
        "![ITL p95 vs load](../results/charts/itl_p95_vs_load.png)",
        "",
        "![Cost per 1M output tokens](../results/charts/cost_per_million_tokens.png)",
        "",
        "## What the results show",
        "",
    ]
    lines += findings_text(headline, by_run)
    lines += [
        "",
        "![Throughput vs concurrency](../results/charts/throughput_vs_concurrency.png)",
        "",
        "![ITL distribution](../results/charts/itl_distribution.png)",
        "",
        "## Answer quality (GSM8K)",
        "",
    ]
    lines += quality_text(headline)
    lines += ["", "## Per-configuration results", ""]
    for run in runs:
        s = run.config["setup"]
        gpu = (run.env.get("gpu") or [{}])[0]
        lines += [
            f"### {run.label} (`{run.scenario}`)",
            "",
            f"- **Hardware:** {gpu.get('name')} ({gpu.get('memory')}, driver {gpu.get('driver')}); "
            f"{s['hardware']}",
            f"- **Engine:** {s['engine']} `{s['server_args']}`",
            f"- **Model:** `{s['model_file']}`",
            f"- **Run:** [`{run.run_id}`](../results/{run.run_id}/), code "
            f"`{run.env['git']['sha'][:7]}`"
            + (
                " (recorded `dirty: true`: untracked files only, findings F10)"
                if run.env["git"]["dirty"]
                else ""
            )
            + f"; rates from probe `{(run.env.get('rates_from_probe') or {}).get('probe_run', '-')}`"
            + (
                f" (1-{max(int(p['level']) for p in probes[run.run_id])} users)"
                if run.run_id in probes
                else ""
            ),
            "",
            level_table(by_run[run.run_id]),
        ]
    worst_lag = max(lv["scheduling_lag_p99"] for rows in by_run.values() for lv in rows)
    lines += method_text(worst_lag) + limitations_text()
    return "\n".join(lines).rstrip("\n") + "\n"


def findings_text(headline, by_run) -> list[str]:
    by = {h["label"]: h for h in headline}
    out = []
    fp16, awq, big = by.get("1.5B FP16"), by.get("1.5B AWQ 4-bit"), by.get("7B AWQ 4-bit")
    if fp16 and awq:
        light_fp16 = min(by_run[fp16["run"].run_id], key=lambda r: r["level"])["tpot_p50"]
        light_awq = min(by_run[awq["run"].run_id], key=lambda r: r["level"])["tpot_p50"]
        out.append(
            f"- **4-bit AWQ is faster than FP16 on the same 1.5B model, most of all for a single "
            f"stream:** {light_awq * 1000:.1f} vs {light_fp16 * 1000:.1f} ms per output token at the "
            f"lightest load ({light_fp16 / light_awq:.1f}x), but {awq['ok']['throughput_rps']:.2f} vs "
            f"{fp16['ok']['throughput_rps']:.2f} req/s at the SLO limit "
            f"({awq['ok']['throughput_rps'] / fp16['ok']['throughput_rps']:.2f}x). One stream is "
            "limited by memory bandwidth (reading the weights, where 4-bit helps most); large "
            "batches make the GPU compute-bound, where AWQ also pays to dequantise."
        )
    if awq and big:
        ratio = big["cost"] / awq["cost"] if big["cost"] and awq["cost"] else None
        out.append(
            f"- **Model size dominates cost:** 7B AWQ meets the SLO up to "
            f"{big['ok']['throughput_rps']:.2f} req/s, "
            + (f"about {ratio:.0f}x the cost per token of 1.5B AWQ" if ratio else "")
            + "; its inter-token latency is what breaks first."
        )
    out.append(
        "- **Past saturation, queues explode:** the open loop keeps sending at the offered rate, so "
        "once a configuration cannot keep up, waiting requests pile up and TTFT climbs by orders of "
        "magnitude (see the highest rates in the tables). A closed-loop test would have hidden "
        "this, because its clients slow down with the server."
    )
    return out


def quality_text(headline) -> list[str]:
    rows = [h for h in headline if h["quality"]]
    if not rows:
        return ["No GSM8K results yet."]
    out = [
        "The same 200 GSM8K test questions (fixed seed) for every configuration, zero-shot, "
        "temperature 0, at most 1024 output tokens, scored by exact final number "
        "(`bench/quality/gsm8k.py`). Zero-shot scores compare these configurations with each "
        "other; they are not comparable with published few-shot scores.",
        "",
        "| Configuration | Correct | Accuracy | 95% Wilson interval | Errors | Truncated |",
        "|---|---|---|---|---|---|",
    ]
    for h in rows:
        q = h["quality"]
        low, high = wilson_interval(q["correct"], q["questions"])
        out.append(
            f"| {h['label']} | {q['correct']}/{q['questions']} | {q['accuracy']:.1%} | "
            f"{low:.1%}-{high:.1%} | {q['errors']} | {q['truncated']} |"
        )
    pairs = [("1.5B FP16", "1.5B AWQ 4-bit"), ("7B AWQ 4-bit", "1.5B AWQ 4-bit")]
    by = {h["label"]: h for h in rows}
    out += ["", "Paired comparison on the same questions (exact McNemar test):", ""]
    for a, b in pairs:
        if a in by and b in by:
            r = mcnemar(_answers(by[a]["quality"]["run_id"]), _answers(by[b]["quality"]["run_id"]))
            verdict = "significant" if r["p_value"] < 0.05 else "not significant at 0.05"
            p = "< 0.0001" if r["p_value"] < 0.0001 else f"= {r['p_value']:.3f}"
            out.append(
                f"- **{a} vs {b}:** only {a} right on {r['only_a_right']} questions, only {b} "
                f"right on {r['only_b_right']}: p {p} ({verdict})."
            )
    return out


def _answers(run_id: str) -> dict[int, bool]:
    import pyarrow.parquet as pq

    table = pq.read_table(RESULTS / "quality" / run_id / "answers.parquet")
    return {r["index"]: r["correct"] for r in table.to_pylist()}


def method_text(worst_lag_s: float) -> list[str]:
    return [
        "## Method",
        "",
        "- **Path:** client -> Relay (standalone: one key, in-memory limits) -> vLLM on the same "
        "VM. Relay's own added latency was measured separately (about 4 ms at p50 before the first "
        "token on the dev laptop; `docs/findings.md` F2).",
        "- **Load:** open loop, Poisson arrivals at 5 rates from 0.25x to 1.25x of a closed-loop "
        "probe's peak (probe range listed per configuration), 120 s per rate after a 20 s warm-up "
        "that is excluded.",
        "- **Workload:** identical for every configuration: prompt lengths log-normal (median 200 "
        "tokens, 16-600), output lengths log-normal (median 96, 16-256) with `ignore_eos`, seed 42; "
        "prompt openings salted per run and level so no prompt is ever resent (prefix caching would "
        "flatter TTFT; findings F1).",
        "- **Metrics:** TTFT = first content token - send; ITL = gap between token arrivals; "
        "TPOT = (end - first token) / (tokens - 1); percentiles by linear interpolation; rates "
        "over the window from the end of warm-up to the last response.",
        "- **Validity checks:** every request served by the expected backend; host RAM never "
        "under 1 GiB; requests sent no more than 100 ms late (p99), so the client was never the "
        "bottleneck. Runs failing a check are set aside, not reported. The harness checks these "
        "automatically (the client-lag check was added after the 7B run); in the reported runs the "
        f"worst scheduling lag p99 at any level is {worst_lag_s * 1000:.0f} ms.",
        "",
    ]


def limitations_text() -> list[str]:
    return [
        "## Limitations",
        "",
        "- **One GPU class:** a free Colab T4 (Turing, 16 GB). Numbers do not transfer to other GPUs "
        "and are not capacity planning for larger hardware.",
        "- **Assumed prices:** cost is measured throughput x a public list price, not a bill.",
        "- **One engine on the GPU:** vLLM 0.31.0. llama.cpp is Relay's development backend but was "
        "not benchmarked: the laptop proved an unreliable benchmark host (ADR-001 amendment).",
        "- **Synthetic prompts:** lengths follow documented distributions; content is filler text. "
        "Real traffic with shared system prompts would gain from prefix caching.",
        "- **Single runs:** one run per configuration (the 1.5B configurations agreed with an "
        "earlier session where rates overlapped, findings F11); the SLO limit is a bracket between "
        "two tested rates, not a point.",
        "- **Zero-shot GSM8K on 200 questions:** about +-7 points of uncertainty per configuration.",
    ]


def main() -> None:
    print(f"wrote {generate()}")
    for p in sorted(CHARTS.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
