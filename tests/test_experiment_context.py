"""experiment/context.py has never had a dedicated test file - this one
is scoped to stage 6's new addition (the optional `ohlcv` merge), plus a
baseline sanity check of the pre-existing behavior it must not disturb.
"""
from datetime import datetime, timedelta, timezone

from market_agent.experiment.context import build_context
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FlatPrices(PriceSeriesProvider):
    def close_price(self, ticker, as_of):
        return 100.0


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff]


def _bar(i, o, h, l, c, v=1_000_000):
    return Bar(date=START + timedelta(days=i), open=o, high=h, low=l, close=c, volume=v)


def test_build_context_without_ohlcv_never_adds_technical_fields():
    ctx = build_context(FlatPrices(), "ACME", START + timedelta(days=30))
    assert "breakout_state" not in ctx.extra
    assert "trend_direction" not in ctx.extra
    assert "prior_return_bucket" in ctx.extra  # pre-existing behavior, unaffected


def test_build_context_with_ohlcv_merges_flat_technical_state_fields():
    ohlcv = FakeOHLCV()
    bars = [_bar(i, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(30)]
    ohlcv.set_bars("ACME", bars)
    ohlcv.set_bars("SPY", bars)
    ctx = build_context(FlatPrices(), "ACME", bars[-1].date, benchmark_ticker="SPY", ohlcv=ohlcv)
    assert ctx.extra["trend_direction"] == "UP"
    assert "breakout_state" in ctx.extra
    assert ctx.to_dict()["trend_direction"] == "UP"  # flat, top-level - matchable by _condition_matches_context
