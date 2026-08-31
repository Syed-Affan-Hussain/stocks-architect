from datetime import datetime, timezone

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.retrieval.regime import classify_regime


class FakePrices(PriceSeriesProvider):
    def __init__(self, prices):
        self.prices = prices

    def close_price(self, ticker, as_of):
        return self.prices.get((ticker, as_of.date().isoformat()))


NOW = datetime(2024, 3, 1, tzinfo=timezone.utc)


def test_classify_risk_off_on_large_trailing_decline():
    prices = FakePrices({("SPY", "2024-01-01"): 400.0, ("SPY", "2024-03-01"): 350.0})  # -12.5%
    assert classify_regime(prices, NOW) == "RISK_OFF"


def test_classify_risk_on_on_large_trailing_gain():
    prices = FakePrices({("SPY", "2024-01-01"): 400.0, ("SPY", "2024-03-01"): 460.0})  # +15%
    assert classify_regime(prices, NOW) == "RISK_ON"


def test_classify_normal_for_modest_move():
    prices = FakePrices({("SPY", "2024-01-01"): 400.0, ("SPY", "2024-03-01"): 410.0})  # +2.5%
    assert classify_regime(prices, NOW) == "NORMAL"


def test_classify_unknown_when_price_data_missing():
    prices = FakePrices({})
    assert classify_regime(prices, NOW) == "UNKNOWN"
