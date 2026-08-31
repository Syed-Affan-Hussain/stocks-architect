"""Builds a real, point-in-time ContextSnapshot for a real event - the
live-data counterpart to the hand-constructed ContextSnapshots used in
Stage 1's tests. Every value here is computed using only price data on or
before `as_of` (and, for the two episodic-history fields, only
episodic_events rows published before `as_of` if a `conn` is supplied);
nothing here ever looks at data published after the event it's
describing.

STAGE 6: technical concept fields (concepts/technical_context.py) are
merged in ONLY when an `ohlcv` provider is explicitly passed - this
parameter defaults to None specifically so every existing caller/test
double that only ever had a PriceSeriesProvider (close-price point
lookups) keeps working completely unchanged; it simply never gets
technical fields, same "disclosed absence, not a fabricated value"
convention already used for sector_momentum/event_surprise. Fields are
merged as FLAT top-level keys via ContextSnapshot's `extra` dict - the
same mechanism prior_return_bucket already uses - specifically so
agents/adaptive_agent.py's `_condition_matches_context` (plain dict-key
equality) can match a technical-concept-conditioned relationship with
zero changes to that function.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from market_agent.concepts.technical_context import (
    TECHNICAL_STATE_FIELD_NAMES, TechnicalMarketContext, build_technical_context,
)
from market_agent.events.schema import ContextSnapshot
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.retrieval.regime import classify_regime
from market_agent.retrieval.similarity import prior_return_bucket

REALIZED_VOL_SAMPLE_DAYS = (0, 5, 10, 15, 20)  # weekly-spaced point samples over the 20d window


def _technical_state_fields(ctx: TechnicalMarketContext) -> dict:
    d = ctx.to_dict()
    return {name: d[name] for name in TECHNICAL_STATE_FIELD_NAMES}


def _trailing_return(prices: PriceSeriesProvider, ticker: str, as_of: datetime, lookback_days: int) -> float | None:
    end_price = prices.close_price(ticker, as_of)
    start_price = prices.close_price(ticker, as_of - timedelta(days=lookback_days))
    if end_price is None or start_price is None or start_price <= 0:
        return None
    return end_price / start_price - 1.0


def _realized_vol_20d(prices: PriceSeriesProvider, ticker: str, as_of: datetime) -> float | None:
    """A DISCLOSED APPROXIMATION, not true daily realized volatility:
    PriceSeriesProvider only exposes point lookups (close_price), not a
    full return series - extending that interface would break every
    existing test double built against it (Stage 1-3). This samples 5
    weekly-spaced closes over the trailing 20 days and takes the stdev of
    the resulting 4 weekly returns - a real, computed number, but a
    coarser one than true daily realized vol would give. Good enough to
    be a candidate explanatory variable (this stage's own framing: "do
    not assume these variables contain signal"), not claimed to be more
    precise than it is."""
    closes = [prices.close_price(ticker, as_of - timedelta(days=d)) for d in REALIZED_VOL_SAMPLE_DAYS]
    if any(c is None or c <= 0 for c in closes):
        return None
    closes = list(reversed(closes))  # oldest -> newest
    weekly_returns = [closes[i + 1] / closes[i] - 1.0 for i in range(len(closes) - 1)]
    if len(weekly_returns) < 2:
        return None
    mean = sum(weekly_returns) / len(weekly_returns)
    variance = sum((r - mean) ** 2 for r in weekly_returns) / (len(weekly_returns) - 1)
    return variance ** 0.5


def _episodic_history_fields(conn: sqlite3.Connection | None, entity: str, event_type: str,
                              as_of: datetime) -> tuple[float | None, int | None]:
    """(days_since_last_same_entity_event, competing_events_same_day) -
    both None if no `conn` was supplied (e.g. building context before any
    episodic history exists to query)."""
    if conn is None:
        return None, None
    as_of_iso = as_of.isoformat()
    last_same_entity = conn.execute(
        """SELECT published_at FROM episodic_events WHERE entity = ? AND event_type = ? AND published_at < ?
           ORDER BY published_at DESC LIMIT 1""",
        (entity, event_type, as_of_iso),
    ).fetchone()
    days_since = None
    if last_same_entity is not None:
        last_dt = datetime.fromisoformat(last_same_entity["published_at"])
        days_since = (as_of - last_dt).total_seconds() / 86400.0

    same_day_start = as_of.date().isoformat() + "T00:00:00"
    same_day_end = as_of.date().isoformat() + "T23:59:59"
    competing = conn.execute(
        """SELECT COUNT(DISTINCT entity) c FROM episodic_events
           WHERE entity != ? AND published_at >= ? AND published_at <= ?""",
        (entity, same_day_start, same_day_end),
    ).fetchone()
    return days_since, competing["c"] if competing is not None else None


def build_context(prices: PriceSeriesProvider, entity: str, as_of: datetime, benchmark_ticker: str = "SPY",
                   event_type: str = "GUIDANCE_CHANGE", conn: sqlite3.Connection | None = None,
                   ohlcv: OHLCVProvider | None = None) -> ContextSnapshot:
    regime = classify_regime(prices, as_of, benchmark_ticker)
    prior_1d = _trailing_return(prices, entity, as_of, 1)
    prior_5d = _trailing_return(prices, entity, as_of, 5)
    prior_20d = _trailing_return(prices, entity, as_of, 20)
    prior_60d = _trailing_return(prices, entity, as_of, 60)
    realized_vol = _realized_vol_20d(prices, entity, as_of)
    market_return_20d = _trailing_return(prices, benchmark_ticker, as_of, 20)
    days_since, competing = _episodic_history_fields(conn, entity, event_type, as_of)

    # Stored as a bucket LABEL (not the raw float) specifically so a validated_relationship's
    # condition_json can match on it directly (learn/hypothesis.py proposes conditions like
    # {"prior_return_bucket": "LARGE_DECLINE"} - matching that against a raw float would need
    # re-deriving the bucket at every match check instead of once, here, at context-build time).
    extra = {"prior_return_bucket": prior_return_bucket(prior_5d)}
    if ohlcv is not None:
        technical = build_technical_context(ohlcv, entity, as_of, benchmark_ticker)
        extra.update(_technical_state_fields(technical))

    # sector_momentum, event_surprise, earnings_proximity: no data source wired in - see
    # ContextSnapshot's own class docstring for why these stay disclosed placeholders rather than
    # fabricated values.
    return ContextSnapshot(
        regime=regime, prior_5d_return=prior_5d, sector_momentum="UNKNOWN",
        prior_1d_return=prior_1d, prior_20d_return=prior_20d, prior_60d_return=prior_60d,
        realized_vol_20d=realized_vol, market_return_20d=market_return_20d,
        published_weekday=as_of.weekday(), published_hour_utc=as_of.hour,
        days_since_last_same_entity_event=days_since, competing_events_same_day=competing,
        extra=extra,
    )
