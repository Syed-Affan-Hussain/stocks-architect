from datetime import datetime, timedelta, timezone

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider
from market_agent.research.historical_reaction import compute_historical_reaction, open_historical_ledger
from market_agent.research.market_context import build_market_context
from market_agent.store import db

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff]


class FakePrices(PriceSeriesProvider):
    def __init__(self):
        self.data = {}

    def set_price(self, ticker, date, price):
        self.data[(ticker, date.date().isoformat())] = price

    def close_price(self, ticker, as_of):
        candidates = [(d, p) for (t, d), p in self.data.items() if t == ticker and d <= as_of.date().isoformat()]
        return max(candidates, key=lambda dp: dp[0])[1] if candidates else None


def _bar(i, price):
    return Bar(date=START + timedelta(days=i), open=price, high=price * 1.01, low=price * 0.99, close=price,
               volume=1_000_000)


def test_market_context_none_when_no_price_history():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    assert build_market_context(ohlcv, prices, "GHOST", as_of=START) is None


def test_market_context_real_return_and_narrative():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    bars = [_bar(i, 100 + i) for i in range(100)]
    ohlcv.set_bars("ACME", bars)
    for b in bars:
        prices.set_price("ACME", b.date, b.close)
        prices.set_price("SPY", b.date, 400.0)
    ctx = build_market_context(ohlcv, prices, "ACME", as_of=bars[-1].date)
    assert ctx is not None
    assert ctx.price == bars[-1].close
    assert ctx.return_1m is not None and ctx.return_1m > 0  # rising price series
    assert "risen" in ctx.narrative_text


def test_open_historical_ledger_missing_file_returns_none():
    assert open_historical_ledger("data_cache/does_not_exist_at_all.sqlite") is None


def test_compute_historical_reaction_below_min_n_returns_none():
    conn = db.connect(":memory:")
    assert compute_historical_reaction(conn, "GUIDANCE_CHANGE", "positive", 20) is None
