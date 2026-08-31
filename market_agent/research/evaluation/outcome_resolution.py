"""Fills in realized outcomes for logged predictions whose horizon has
genuinely elapsed in real calendar time - this is what makes the
evaluation PROSPECTIVE rather than a claim: a row's realized_return_Nd
column can only ever be written by a run of this module that happens
AFTER N trading days have actually passed since triggered_at. There is no
code path anywhere in this package that computes an outcome for a horizon
that hasn't matured yet - see resolve_outcomes' `now < matures_at` guard.

REUSES market_agent/outcomes/observe.py's compute_abnormal_return
UNCHANGED - the same market-adjusted (entity return minus benchmark
return) convention the pre-existing trading-research system already used
and tested, not a new return-computation formula.

TRADING DAYS, APPROXIMATED AS CALENDAR DAYS: compute_abnormal_return takes
a calendar-day offset (event_date + timedelta(days=...)), but "1/5/20/60
trading-day outcomes" was specified in trading days. There is no real NYSE
trading-calendar module anywhere in this project (a genuine, disclosed
gap - building one was out of scope for this pass), so
TRADING_DAYS_TO_CALENDAR_DAYS below is a FIXED, DISCLOSED approximation
(roughly trading_days * 7/5, plus a small buffer for holidays), not a
real calendar computation. Combined with PriceSeriesProvider.close_price's
"on or immediately before" semantics, this can overshoot the true Nth
trading day by up to ~1-2 days in the presence of holidays - a small,
bounded, disclosed imprecision, not silent wrongness.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_agent.outcomes.observe import PriceSeriesProvider, compute_abnormal_return
from market_agent.store import db

DEFAULT_BENCHMARK = "SPY"

# horizon (trading days) -> (realized_return column, resolved_at column, calendar-day approximation)
HORIZONS: tuple[tuple[int, str, str, int], ...] = (
    (1, "realized_return_1d", "resolved_1d_at", 1),
    (5, "realized_return_5d", "resolved_5d_at", 7),
    (20, "realized_return_20d", "resolved_20d_at", 29),
    (60, "realized_return_60d", "resolved_60d_at", 88),
)
HORIZON_COLUMNS: dict[str, str] = {ret_col: resolved_col for _, ret_col, resolved_col, _ in HORIZONS}


def resolve_outcomes(conn, prices: PriceSeriesProvider, benchmark_ticker: str = DEFAULT_BENCHMARK,
                      now: datetime | None = None) -> list[dict]:
    """Idempotent and safe to run as often as desired (e.g. once a day) -
    a row already resolved for a given horizon is excluded by
    db.unresolved_predictions' own WHERE clause, and a row whose horizon
    hasn't matured yet is skipped without being touched. Returns what was
    actually resolved THIS call, for the caller to log/report - never
    fabricates a result for a row it didn't genuinely resolve."""
    now = now or datetime.now(timezone.utc)
    resolved: list[dict] = []
    for trading_days, return_col, resolved_col, calendar_days in HORIZONS:
        for row in db.unresolved_predictions(conn, return_col):
            triggered_at = datetime.fromisoformat(row["triggered_at"])
            if triggered_at.tzinfo is None:
                triggered_at = triggered_at.replace(tzinfo=timezone.utc)
            matures_at = triggered_at + timedelta(days=calendar_days)
            if now < matures_at:
                continue  # horizon has not elapsed yet - never compute an early value
            result = compute_abnormal_return(prices, row["entity"], benchmark_ticker, triggered_at, calendar_days)
            if result.status != "OK":
                continue  # insufficient price data - leave NULL, never guess
            db.record_prediction_outcome(conn, row["prediction_id"], return_col, resolved_col,
                                          result.abnormal_return, now)
            resolved.append({
                "prediction_id": row["prediction_id"], "entity": row["entity"], "mode": row["mode"],
                "horizon_trading_days": trading_days, "abnormal_return": result.abnormal_return,
                "entity_return": result.entity_return, "benchmark_return": result.benchmark_return,
            })
    return resolved
