"""Item 12: descriptive price/technical context, built entirely by
REUSING the existing, unmodified concepts/technical_context.py and
retrieval/regime.py - no new indicator logic. Every sentence here
describes what already happened, never what will happen - see the
"CONTEXT, NOT PREDICTION" framing required by this product's philosophy.
"""
from __future__ import annotations

from datetime import datetime, timezone

from market_agent.concepts.technical_context import build_technical_context
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.research.schema import MarketContext
from market_agent.retrieval.regime import classify_regime


def _return_over(prices: PriceSeriesProvider, ticker: str, as_of: datetime, days: int) -> float | None:
    end = prices.close_price(ticker, as_of)
    from datetime import timedelta
    start = prices.close_price(ticker, as_of - timedelta(days=days))
    if end is None or start is None or start <= 0:
        return None
    return end / start - 1.0


def build_market_context(ohlcv: OHLCVProvider, prices: PriceSeriesProvider, ticker: str,
                          benchmark_ticker: str = "SPY", as_of: datetime | None = None) -> MarketContext | None:
    """Returns None (SOURCE_UNAVAILABLE, disclosed by the caller) if there
    is no usable price history for `ticker` at all."""
    as_of = as_of or datetime.now(timezone.utc)
    price = prices.close_price(ticker, as_of)
    if price is None:
        return None

    technical = build_technical_context(ohlcv, ticker, as_of, benchmark_ticker)
    regime = classify_regime(prices, as_of, benchmark_ticker)
    return_1m = _return_over(prices, ticker, as_of, 30)
    return_3m = _return_over(prices, ticker, as_of, 90)

    parts = []
    if return_1m is not None:
        direction = "risen" if return_1m > 0 else "fallen" if return_1m < 0 else "been roughly flat"
        parts.append(f"Shares have {direction} {abs(return_1m):.1%} over the last month")
    if return_3m is not None:
        parts.append(f"{return_3m:+.1%} over the last three months")
    trend_text = {"UP": "an uptrend", "DOWN": "a downtrend", "FLAT": "a flat/range-bound trend",
                  "UNKNOWN": "an undetermined trend (insufficient history)"}[technical.trend_direction]
    vol_text = {"COMPRESSION": "compressed", "EXPANSION": "expanded", "NORMAL": "typical",
                "UNKNOWN": "undetermined"}[technical.volatility_state]
    parts.append(f"Price structure currently reflects {trend_text}, with {vol_text} volatility "
                 f"and an overall market regime classified as {regime}")
    narrative = ". ".join(parts) + "." if parts else "Insufficient price history for a market-context narrative."

    return MarketContext(as_of=as_of.isoformat(), price=price, return_1m=return_1m, return_3m=return_3m,
                          trend_direction=technical.trend_direction, volatility_state=technical.volatility_state,
                          regime=regime, narrative_text=narrative)
