"""Item 13: DESCRIPTIVE historical event-reaction statistics, reusing the
EXISTING event/outcome ledger (market_agent.store.db, episodic_events)
this project already built and populated with real SEC EDGAR guidance-
change events and real market-adjusted abnormal returns across many
companies (stages 1-7's trading-research system).

THE QUESTION IS DELIBERATELY DIFFERENT FROM THE TRADING-RESEARCH SYSTEM'S
OWN QUESTION: that system asked "does this condition have a statistically
and economically significant, TEST-surviving PREDICTIVE edge" (Holm
correction, MIN_N gates, frozen TEST holdouts, ...) - all of that
promotion/validation machinery is IGNORED here on purpose. This module
asks a narrower, purely descriptive question: "historically, across many
real companies, how has the market reacted to this KIND of event" - a
median, a percentage positive, a sample size. No claim of predictive
validity, no promotion, no significance test. Every HistoricalReaction
explicitly says so in its evidence.

CROSS-COMPANY, NOT COMPANY-SPECIFIC: a single company's own event history
is almost always too sparse (a handful of guidance changes over years) to
say anything statistically meaningful on its own - this deliberately pools
across every company in the existing real ledger for the SAME event_type/
direction, which is what "how has the market historically reacted to
SIMILAR developments" actually requires."""
from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

from market_agent.research.schema import HistoricalReaction
from market_agent.store import db

DEFAULT_LEDGER_PATH = "data_cache/stage7_final_report.sqlite"  # the most recent real, populated ledger
#                                                                  from this project's own trading-research
#                                                                  runs - see module docstring. A caller may
#                                                                  pass any other populated ledger explicitly.
MIN_N_FOR_REPORTING = 10  # purely descriptive floor - not the trading system's MIN_N=15 promotion gate,
#                            just "enough to state a median/percentage without it being one or two data points"


def open_historical_ledger(path: str | Path = DEFAULT_LEDGER_PATH) -> sqlite3.Connection | None:
    """Returns None (never raises) if the ledger file doesn't exist -
    callers must treat that as SOURCE_UNAVAILABLE, not fabricate stats."""
    if not Path(path).exists():
        return None
    return db.connect(str(path))


def compute_historical_reaction(conn: sqlite3.Connection, event_type: str, direction: str,
                                 horizon_days: int) -> HistoricalReaction | None:
    """Real, descriptive stats from the existing event/outcome ledger -
    median reaction, % positive, sample size, market-adjusted (abnormal,
    not raw) return. Returns None if there are too few real matching
    cases to report responsibly (below MIN_N_FOR_REPORTING) - never a
    stat computed from a handful of points and presented as if reliable."""
    rows = db.query_events(conn, event_type=event_type, outcome_known_only=True)
    rows = [r for r in rows if r["direction"] == direction and r["horizon_days"] == horizon_days]
    rows = db.deduplicate_by_real_event(rows)
    returns = [r["realized_abnormal_return"] for r in rows]
    n = len(returns)
    if n < MIN_N_FOR_REPORTING:
        return None
    median = statistics.median(returns)
    pct_positive = sum(1 for r in returns if r > 0) / n
    return HistoricalReaction(
        event_type=event_type, direction=direction, horizon_days=horizon_days, n=n, median_reaction=median,
        pct_positive=pct_positive,
        evidence=[f"DESCRIPTIVE, cross-company historical association - NOT a prediction for this specific "
                  f"company. N={n} real, deduplicated {event_type}/{direction} events across the existing "
                  f"historical ledger, {horizon_days}-day market-adjusted (abnormal) reaction.",
                  f"Median reaction {median:+.2%}; positive in {pct_positive:.0%} of cases."])


def historical_reactions_for_recent_event_types(conn: sqlite3.Connection, event_types: list[tuple[str, str]],
                                                  horizon_days_list: list[int] = (1, 5, 20, 60)
                                                  ) -> list[HistoricalReaction]:
    """One HistoricalReaction per (event_type, direction) actually
    observed in THIS company's recent timeline, at each of the standard
    horizons, when the cross-company ledger has enough real cases to say
    something responsible."""
    results: list[HistoricalReaction] = []
    for event_type, direction in event_types:
        for horizon in horizon_days_list:
            r = compute_historical_reaction(conn, event_type, direction, horizon)
            if r is not None:
                results.append(r)
    return results
