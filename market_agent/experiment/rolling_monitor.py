"""Live ADAPTIVE-vs-STATIC rolling monitoring - stage 5 item 9.

DESCRIPTIVE, NOT INFERENTIAL - read this before wiring this into anything
that makes a decision. A rolling window's metrics are the plain,
unadjusted, most-recent-N-observations comparison - no significance test,
no confidence interval, no multiple-testing correction. They exist to
answer "what does recent performance look like", not "is ADAPTIVE
statistically better than STATIC" - that question is answered only by
learn/hypothesis_testing.py's governed test (min-N, held-out, Holm-
corrected) or experiment/chronological_eval.py's before/after windows,
never by this module. Every RollingComparison below carries an explicit
`is_descriptive_only=True` marker for exactly this reason - so a caller
cannot accidentally treat "ADAPTIVE improved over the last 50" as
evidence of anything on its own.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from market_agent.experiment.metrics import MetricsReport, PredictionOutcome, compute_metrics

STANDARD_WINDOW_SIZES = (50, 100, 250, 500)


@dataclass
class RollingComparison:
    window_size: int
    n_available: int  # actual N used - may be less than window_size if fewer resolved predictions exist
    static_metrics: MetricsReport
    adaptive_metrics: MetricsReport
    is_descriptive_only: bool = True


@dataclass
class RollingMonitorReport:
    dimension: str          # "overall" | "event_type" | "horizon" | "direction" | "regime" | "confidence_bucket"
    dimension_value: str | None
    windows: list[RollingComparison] = field(default_factory=list)


def _resolved_rows(conn: sqlite3.Connection, agent: str, extra_clause: str = "", extra_params: tuple = ()
                    ) -> list[sqlite3.Row]:
    query = (f"SELECT * FROM episodic_events WHERE model_version LIKE ? AND outcome_locked = 1 "
             f"AND realized_abnormal_return IS NOT NULL {extra_clause} ORDER BY published_at DESC")
    return conn.execute(query, (f"{agent}%",) + extra_params).fetchall()


def _rows_to_outcomes(rows: list[sqlite3.Row]) -> list[PredictionOutcome]:
    return [PredictionOutcome(r["predicted_impact"], r["predicted_confidence"], r["realized_abnormal_return"])
            for r in rows]


def rolling_comparison(conn: sqlite3.Connection, extra_clause: str = "", extra_params: tuple = (),
                        window_sizes: tuple[int, ...] = STANDARD_WINDOW_SIZES) -> list[RollingComparison]:
    """The most recent N resolved predictions per agent, for each window
    size, filtered by `extra_clause`/`extra_params` (a raw SQL fragment -
    used internally by rolling_by_dimension below, not meant to be a
    general injection point for external callers)."""
    static_rows = _resolved_rows(conn, "STATIC", extra_clause, extra_params)
    adaptive_rows = _resolved_rows(conn, "ADAPTIVE", extra_clause, extra_params)

    comparisons = []
    for size in window_sizes:
        static_metrics = compute_metrics(_rows_to_outcomes(static_rows[:size]))
        adaptive_metrics = compute_metrics(_rows_to_outcomes(adaptive_rows[:size]))
        comparisons.append(RollingComparison(
            window_size=size, n_available=min(len(static_rows), len(adaptive_rows), size),
            static_metrics=static_metrics, adaptive_metrics=adaptive_metrics,
        ))
    return comparisons


def rolling_by_dimension(conn: sqlite3.Connection, dimension: str,
                          window_sizes: tuple[int, ...] = STANDARD_WINDOW_SIZES) -> list[RollingMonitorReport]:
    """One RollingMonitorReport per distinct value of `dimension` -
    "event_type", "horizon_days", "direction", or a context field like
    "regime" (looked up via json_extract) or "predicted_confidence"
    (the closest existing proxy for a 'confidence bucket', since
    predictions don't carry a separate discretized bucket field)."""
    column_map = {
        "event_type": "event_type", "horizon": "horizon_days", "direction": "direction",
        "regime": "json_extract(context_json, '$.regime')", "confidence_bucket": "predicted_confidence",
    }
    if dimension not in column_map:
        raise ValueError(f"Unknown dimension {dimension!r} - must be one of {list(column_map)}")
    column_expr = column_map[dimension]

    values = [r[0] for r in conn.execute(
        f"SELECT DISTINCT {column_expr} FROM episodic_events WHERE {column_expr} IS NOT NULL").fetchall()]

    reports = []
    for value in values:
        comparisons = rolling_comparison(conn, extra_clause=f"AND {column_expr} = ?", extra_params=(value,),
                                          window_sizes=window_sizes)
        reports.append(RollingMonitorReport(dimension=dimension, dimension_value=str(value), windows=comparisons))
    return reports
