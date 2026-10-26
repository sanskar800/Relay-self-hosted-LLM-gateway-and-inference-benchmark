-- Benchmark analysis queries (DuckDB), run by bench/analysis/analyze.py.
--
-- Every metric is recomputed from the RAW per-request timestamps in results.parquet, not
-- from the harness's summary.json: two independent implementations of the same definitions
-- (bench/loadgen/metrics.py), which tests/unit/test_analysis.py checks against each other.
--
-- Inputs registered by analyze.py:
--   requests  one row per request: results/run-*/results.parquet
--   runs      one row per run: run_id, scenario, warmup_s, slo_ttft_s, slo_tpot_s
--   answers   one row per GSM8K answer: results/quality/gsm8k-*/answers.parquet (+ run_id)
-- Percentiles use linear interpolation (quantile_cont), as metrics.percentile does.

-- name: levels
-- Per run and load level: what was offered, what was served, and how fast.
WITH measured AS (
    SELECT q.*, r.warmup_s, r.slo_ttft_s, r.slo_tpot_s,
           q.first_token_s - q.sent_s AS ttft,
           CASE WHEN q.completion_tokens >= 2
                THEN (q.end_s - q.first_token_s) / (q.completion_tokens - 1) END AS tpot,
           q.end_s - q.sent_s AS e2e,
           q.sent_s - q.scheduled_s AS lag
    FROM requests q JOIN runs r USING (run_id)
    WHERE NOT q.in_warmup
),
gaps AS (
    -- Inter-token latency: consecutive token arrival gaps of successful requests.
    SELECT run_id, level,
           unnest(list_transform(range(1, len(token_times_s)),
                                 i -> token_times_s[i + 1] - token_times_s[i])) AS gap
    FROM measured WHERE ok
),
itl AS (
    SELECT run_id, level,
           quantile_cont(gap, 0.50) AS itl_p50,
           quantile_cont(gap, 0.95) AS itl_p95,
           quantile_cont(gap, 0.99) AS itl_p99
    FROM gaps GROUP BY run_id, level
)
SELECT
    m.run_id, any_value(m.scenario) AS scenario, m.level,
    count(*) AS sent,
    count(*) FILTER (WHERE m.ok) AS ok,
    count(*) FILTER (WHERE NOT m.ok) AS errors,
    count(*) FILTER (WHERE NOT m.ok) / count(*) AS error_rate,
    max(m.end_s) - any_value(m.warmup_s) AS window_s,
    count(*) / (max(m.end_s) - any_value(m.warmup_s)) AS offered_rps,
    count(*) FILTER (WHERE m.ok) / (max(m.end_s) - any_value(m.warmup_s)) AS throughput_rps,
    coalesce(sum(m.completion_tokens) FILTER (WHERE m.ok), 0)
        / (max(m.end_s) - any_value(m.warmup_s)) AS output_tokens_per_s,
    count(*) FILTER (WHERE m.ok AND m.ttft <= m.slo_ttft_s
                     AND (m.tpot IS NULL OR m.tpot <= m.slo_tpot_s)) / count(*) AS goodput_ratio,
    quantile_cont(m.ttft, 0.50) FILTER (WHERE m.ok) AS ttft_p50,
    quantile_cont(m.ttft, 0.95) FILTER (WHERE m.ok) AS ttft_p95,
    quantile_cont(m.ttft, 0.99) FILTER (WHERE m.ok) AS ttft_p99,
    any_value(i.itl_p50) AS itl_p50,
    any_value(i.itl_p95) AS itl_p95,
    any_value(i.itl_p99) AS itl_p99,
    quantile_cont(m.tpot, 0.50) FILTER (WHERE m.ok) AS tpot_p50,
    quantile_cont(m.tpot, 0.95) FILTER (WHERE m.ok) AS tpot_p95,
    quantile_cont(m.e2e, 0.50) FILTER (WHERE m.ok) AS e2e_p50,
    quantile_cont(m.e2e, 0.95) FILTER (WHERE m.ok) AS e2e_p95,
    quantile_cont(m.lag, 0.99) AS scheduling_lag_p99
FROM measured m LEFT JOIN itl i USING (run_id, level)
GROUP BY m.run_id, m.level
ORDER BY scenario, m.level;

-- name: itl_samples
-- Every inter-token gap at each level (for the ITL distribution chart).
SELECT run_id, scenario, level,
       unnest(list_transform(range(1, len(token_times_s)),
                             i -> token_times_s[i + 1] - token_times_s[i])) AS gap
FROM requests WHERE ok AND NOT in_warmup
ORDER BY scenario, level;

-- name: quality
-- GSM8K accuracy per configuration (same 200 questions for each).
SELECT run_id,
       count(*) AS questions,
       count(*) FILTER (WHERE correct) AS correct,
       count(*) FILTER (WHERE correct) / count(*) AS accuracy,
       count(*) FILTER (WHERE error IS NOT NULL) AS errors,
       count(*) FILTER (WHERE finish_reason = 'length') AS truncated
FROM answers
GROUP BY run_id
ORDER BY run_id;
