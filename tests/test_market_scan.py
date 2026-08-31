from datetime import datetime, timedelta, timezone

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider
from market_agent.setups.market_scan import scan_entity_market_states, scan_universe_market_states

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data: dict[str, list[Bar]] = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff]


class FakePrices(PriceSeriesProvider):
    def __init__(self):
        self.data: dict[tuple[str, str], float] = {}

    def set_price(self, ticker, date, price):
        self.data[(ticker, date.date().isoformat())] = price

    def close_price(self, ticker, as_of):
        candidates = [(d, p) for (t, d), p in self.data.items() if t == ticker and d <= as_of.date().isoformat()]
        if not candidates:
            return None
        return max(candidates, key=lambda dp: dp[0])[1]


def _bar(i, price=100.0, v=1_000_000):
    return Bar(date=START + timedelta(days=i), open=price, high=price * 1.01, low=price * 0.99, close=price,
               volume=v)


def _flat_series(ohlcv, prices, entity, n_days=40, price=100.0):
    bars = [_bar(i, price) for i in range(n_days)]
    ohlcv.set_bars(entity, bars)
    for b in bars:
        prices.set_price(entity, b.date, price)
        prices.set_price("SPY", b.date, 400.0)
    return bars


def test_scan_uses_only_real_bar_dates():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    bars = _flat_series(ohlcv, prices, "ACME", n_days=40)
    real_dates = {b.date.isoformat() for b in bars}

    observations = scan_entity_market_states(ohlcv, prices, "ACME", sample_every_n_bars=1,
                                               as_of_anchor=bars[-1].date)
    assert observations  # non-empty
    for obs in observations:
        assert obs.as_of in real_dates  # never a fabricated date


def test_sample_every_n_bars_subsamples_the_real_calendar():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    bars = _flat_series(ohlcv, prices, "ACME", n_days=40)

    dense = scan_entity_market_states(ohlcv, prices, "ACME", sample_every_n_bars=1, as_of_anchor=bars[-1].date)
    sparse = scan_entity_market_states(ohlcv, prices, "ACME", sample_every_n_bars=5, as_of_anchor=bars[-1].date)
    assert len(sparse) < len(dense)
    assert len(sparse) == len(dense[::5])


def test_scan_computes_real_regime_and_technical_context():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    bars = _flat_series(ohlcv, prices, "ACME", n_days=40)
    observations = scan_entity_market_states(ohlcv, prices, "ACME", sample_every_n_bars=10, as_of_anchor=bars[-1].date)
    assert observations
    last = observations[-1]
    assert last.regime in ("RISK_ON", "RISK_OFF", "NORMAL", "UNKNOWN")
    assert last.technical.entity == "ACME"


def test_no_history_produces_no_observations():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    observations = scan_entity_market_states(ohlcv, prices, "GHOST", as_of_anchor=START)
    assert observations == []


def test_scan_universe_combines_and_sorts_multiple_entities():
    ohlcv, prices = FakeOHLCV(), FakePrices()
    bars_a = _flat_series(ohlcv, prices, "A", n_days=20)
    bars_b = _flat_series(ohlcv, prices, "B", n_days=20)
    anchor = max(bars_a[-1].date, bars_b[-1].date)

    observations = scan_universe_market_states(ohlcv, prices, ["A", "B"], sample_every_n_bars=1, as_of_anchor=anchor)
    entities_seen = {o.entity for o in observations}
    assert entities_seen == {"A", "B"}
    as_of_values = [o.as_of for o in observations]
    assert as_of_values == sorted(as_of_values)  # chronologically sorted across entities
