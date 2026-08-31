"""Adapts prediction_log rows into the PRE-EXISTING, tested metrics
machinery (market_agent/experiment/metrics.py's compute_metrics,
portfolio_metrics.py's compute_portfolio_metrics) - no new statistics are
computed here, only the data-shape conversion from this package's own log
format to those modules' existing dataclasses.

EVERY REPORT BELOW IS HONEST ABOUT n=0: with zero real prospective
observations matured (see outcome_resolution.py - a horizon's outcome
column is only ever populated after that many real trading days have
actually elapsed), every function here returns a report whose evidence
list SAYS SO explicitly, and compute_metrics/compute_portfolio_metrics'
own existing "too few observations" guards do the rest. Nothing here
fabricates a metric from an empty or near-empty sample.
"""
from __future__ import annotations

import json
import sqlite3

from market_agent.experiment.metrics import MetricsReport, PredictionOutcome, compute_metrics
from market_agent.experiment.portfolio_metrics import PortfolioMetricsReport, TradeRecord, compute_portfolio_metrics
from market_agent.research.evaluation.decision_mapping import confidence_float_to_bucket
from market_agent.research.evaluation.modes import MODES
from market_agent.research.evaluation.outcome_resolution import HORIZONS
from market_agent.store import db

HORIZON_DAYS_TO_COLUMN: dict[int, str] = {days: col for days, col, _, _ in HORIZONS}


def _resolved_rows(conn: sqlite3.Connection, mode: str, horizon_trading_days: int) -> list[sqlite3.Row]:
    if horizon_trading_days not in HORIZON_DAYS_TO_COLUMN:
        raise ValueError(f"Unsupported horizon: {horizon_trading_days} - must be one of {list(HORIZON_DAYS_TO_COLUMN)}")
    return_col = HORIZON_DAYS_TO_COLUMN[horizon_trading_days]
    return [r for r in db.all_predictions(conn, mode=mode) if r[return_col] is not None]


def _to_prediction_outcomes(rows: list[sqlite3.Row], return_col: str) -> list[PredictionOutcome]:
    return [PredictionOutcome(predicted_impact=r["predicted_impact"],
                               predicted_confidence=confidence_float_to_bucket(r["predicted_confidence"]),
                               realized_abnormal_return=r[return_col]) for r in rows]


def _to_trade_records(rows: list[sqlite3.Row], return_col: str, horizon_trading_days: int) -> list[TradeRecord]:
    return [TradeRecord(entity=r["entity"], triggered_at=r["triggered_at"], horizon_days=horizon_trading_days,
                         predicted_impact=r["predicted_impact"], realized_abnormal_return=r[return_col])
            for r in rows if r["predicted_impact"] is not None]  # portfolio metrics need a real direction to trade


def mode_report(conn: sqlite3.Connection, mode: str, horizon_trading_days: int
                 ) -> tuple[MetricsReport, PortfolioMetricsReport]:
    return_col = HORIZON_DAYS_TO_COLUMN[horizon_trading_days]
    rows = _resolved_rows(conn, mode, horizon_trading_days)
    metrics = compute_metrics(_to_prediction_outcomes(rows, return_col))
    portfolio = compute_portfolio_metrics(_to_trade_records(rows, return_col, horizon_trading_days))
    return metrics, portfolio


def compare_modes(conn: sqlite3.Connection, horizon_trading_days: int) -> dict[str, tuple[MetricsReport, PortfolioMetricsReport]]:
    return {mode: mode_report(conn, mode, horizon_trading_days) for mode in MODES}


def _news_state_of(row: sqlite3.Row) -> dict | None:
    snapshot = json.loads(row["inputs_snapshot_json"])
    return snapshot.get("news_state")


def breakdown_by_contradiction(conn: sqlite3.Connection, mode: str, horizon_trading_days: int
                                ) -> dict[str, MetricsReport]:
    """Splits resolved predictions into "news state showed a contradiction
    on at least one axis" vs "no contradiction" at decision time - answers
    "does the model do worse specifically when its own news evidence
    disagreed with itself." Only meaningful for modes that carry a
    news_state (B/C); mode A's snapshot always HAS one recorded (for
    audit) even though mode A's own decision never used it."""
    return_col = HORIZON_DAYS_TO_COLUMN[horizon_trading_days]
    rows = _resolved_rows(conn, mode, horizon_trading_days)
    with_contradiction, without = [], []
    for r in rows:
        ns = _news_state_of(r)
        bucket = with_contradiction if (ns and ns.get("contradiction_axes")) else without
        bucket.append(r)
    return {
        "CONTRADICTION_PRESENT": compute_metrics(_to_prediction_outcomes(with_contradiction, return_col)),
        "NO_CONTRADICTION": compute_metrics(_to_prediction_outcomes(without, return_col)),
    }


def breakdown_by_dominant_axis(conn: sqlite3.Connection, mode: str, horizon_trading_days: int
                                ) -> dict[str, MetricsReport]:
    """Splits by which IMPLICATION_AXES had the single largest-magnitude
    value in the news_state at decision time - answers "does the model do
    better when the dominant news signal was, say, guidance vs. demand."
    A prediction whose news_state carried no signal on any axis is
    excluded from every bucket (nothing to attribute it to)."""
    return_col = HORIZON_DAYS_TO_COLUMN[horizon_trading_days]
    rows = _resolved_rows(conn, mode, horizon_trading_days)
    by_axis: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        ns = _news_state_of(r)
        dims = {k: v for k, v in (ns.get("dimensions", {}) if ns else {}).items() if v is not None}
        if not dims:
            continue
        dominant = max(dims, key=lambda k: abs(dims[k]))
        by_axis.setdefault(dominant, []).append(r)
    return {axis: compute_metrics(_to_prediction_outcomes(axis_rows, return_col))
            for axis, axis_rows in by_axis.items()}


def breakdown_by_magnitude(conn: sqlite3.Connection, mode: str, horizon_trading_days: int
                            ) -> dict[str, MetricsReport]:
    """Splits by |predicted_impact|: SMALL (<0.4), MEDIUM (0.4-0.7), LARGE
    (>0.7) - answers "does a more confident/larger-magnitude call actually
    perform better," the most basic sanity check a magnitude-aware
    quantifier should be able to pass."""
    return_col = HORIZON_DAYS_TO_COLUMN[horizon_trading_days]
    rows = _resolved_rows(conn, mode, horizon_trading_days)
    buckets: dict[str, list[sqlite3.Row]] = {"SMALL": [], "MEDIUM": [], "LARGE": []}
    for r in rows:
        impact = r["predicted_impact"]
        if impact is None:
            continue
        magnitude = abs(impact)
        label = "SMALL" if magnitude < 0.4 else ("MEDIUM" if magnitude < 0.7 else "LARGE")
        buckets[label].append(r)
    return {label: compute_metrics(_to_prediction_outcomes(bucket_rows, return_col))
            for label, bucket_rows in buckets.items()}
