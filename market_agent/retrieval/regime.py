"""Minimal, deterministic regime classification - fills in the `regime`
field ContextSnapshot has carried since Stage 1 (previously hardcoded in
tests) with a real, point-in-time computation. Simple threshold rule on
the benchmark's own trailing return, per the first feasibility dossier's
own recommendation against complex ML regime detection for this kind of
low-sample problem - not a new architectural component, just the first
real implementation of a field that already existed in the schema.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from market_agent.outcomes.observe import PriceSeriesProvider

RISK_OFF_THRESHOLD = -0.10   # trailing 60-trading-day benchmark return below this -> RISK_OFF
RISK_ON_THRESHOLD = 0.10     # above this -> RISK_ON
LOOKBACK_DAYS = 60


def classify_regime(prices: PriceSeriesProvider, as_of: datetime, benchmark_ticker: str = "SPY") -> str:
    end_price = prices.close_price(benchmark_ticker, as_of)
    start_price = prices.close_price(benchmark_ticker, as_of - timedelta(days=LOOKBACK_DAYS))
    if end_price is None or start_price is None or start_price <= 0:
        return "UNKNOWN"
    trailing_return = end_price / start_price - 1.0
    if trailing_return < RISK_OFF_THRESHOLD:
        return "RISK_OFF"
    if trailing_return > RISK_ON_THRESHOLD:
        return "RISK_ON"
    return "NORMAL"
