"""Stage 8: continuous market-state scanning - the "RAW MARKET DATA ->
MARKET STATE -> TECHNICAL STRUCTURE" stages of the pipeline made concrete
and CONTINUOUS, decoupled from any discrete triggering event.

Stages 1-7 are event-centric: a real SEC filing (or dividend/guidance
change) triggers interpretation, then a prediction. Everything about
"what was the technical state on this day" already exists and works for
ANY date (concepts/technical_context.py's build_technical_context takes an
arbitrary `as_of`; retrieval/regime.py's classify_regime does too) - what's
been missing is simply WALKING THE CALENDAR to produce a stream of (entity,
date) observations that aren't anchored to a news event at all. That's all
this module does; it invents no new market-state primitive.

REAL TRADING DAYS ONLY, NEVER A FABRICATED CALENDAR: the scan dates for an
entity are taken directly from that entity's own real cached OHLCV bar
index (via OHLCVProvider.bars), not generated from a weekday/holiday rule
that could land on a day with no real price. `sample_every_n_bars` (fixed,
disclosed, not tuned) subsamples that REAL index by stride to keep a
multi-year, multi-entity scan computationally bounded - see
setups/setup_discovery.py for why an exhaustive daily x every-entity scan
would make the downstream combinatorial search intractable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from market_agent.concepts.technical_context import TechnicalMarketContext, build_technical_context
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.retrieval.regime import classify_regime

DEFAULT_SAMPLE_EVERY_N_BARS = 5   # fixed, disclosed - roughly weekly, not daily; see module docstring
FULL_HISTORY_LOOKBACK_DAYS = 4000  # enough calendar days to cover this project's entire cached range


@dataclass(frozen=True)
class MarketStateObservation:
    entity: str
    as_of: str          # ISO date, taken from a REAL bar in the entity's own cached history
    regime: str          # RISK_ON | RISK_OFF | NORMAL | UNKNOWN (retrieval/regime.py)
    technical: TechnicalMarketContext


def _real_trading_days(ohlcv: OHLCVProvider, entity: str, as_of_anchor: datetime) -> list[datetime]:
    bars = ohlcv.bars(entity, as_of_anchor, lookback_days=FULL_HISTORY_LOOKBACK_DAYS)
    return [b.date for b in sorted(bars, key=lambda b: b.date)]


def scan_entity_market_states(ohlcv: OHLCVProvider, prices: PriceSeriesProvider, entity: str,
                               benchmark_ticker: str = "SPY",
                               sample_every_n_bars: int = DEFAULT_SAMPLE_EVERY_N_BARS,
                               as_of_anchor: datetime | None = None) -> list[MarketStateObservation]:
    """One MarketStateObservation per sampled REAL trading day in
    `entity`'s own cached OHLCV history - each built purely from bars
    strictly on-or-before that day (build_technical_context's own
    point-in-time discipline; nothing here looks ahead)."""
    anchor = as_of_anchor or datetime.now(timezone.utc)
    trading_days = _real_trading_days(ohlcv, entity, anchor)
    sampled_days = trading_days[::sample_every_n_bars]

    observations = []
    for day in sampled_days:
        technical = build_technical_context(ohlcv, entity, day, benchmark_ticker)
        regime = classify_regime(prices, day, benchmark_ticker)
        observations.append(MarketStateObservation(entity=entity, as_of=day.isoformat(), regime=regime,
                                                     technical=technical))
    return observations


def scan_universe_market_states(ohlcv: OHLCVProvider, prices: PriceSeriesProvider, entities: list[str],
                                 benchmark_ticker: str = "SPY",
                                 sample_every_n_bars: int = DEFAULT_SAMPLE_EVERY_N_BARS,
                                 as_of_anchor: datetime | None = None) -> list[MarketStateObservation]:
    """Scans every entity in `entities` independently (each entity's OWN
    real trading-day calendar - different entities can have different
    listing histories) and returns one combined, chronologically sorted
    stream. Missing/delisted-entity histories degrade to an empty
    contribution for that entity, never an error for the whole scan."""
    all_observations: list[MarketStateObservation] = []
    for entity in entities:
        all_observations.extend(scan_entity_market_states(ohlcv, prices, entity, benchmark_ticker,
                                                            sample_every_n_bars, as_of_anchor))
    all_observations.sort(key=lambda o: o.as_of)
    return all_observations
