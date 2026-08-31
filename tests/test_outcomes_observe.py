from datetime import datetime, timezone

import pytest

from market_agent.outcomes.observe import PriceSeriesProvider, compute_abnormal_return


class FakePriceSeriesProvider(PriceSeriesProvider):
    """Network-free: fixed prices keyed by (ticker, iso-date)."""
    def __init__(self, prices: dict):
        self.prices = prices

    def close_price(self, ticker, as_of):
        return self.prices.get((ticker, as_of.date().isoformat()))


def test_abnormal_return_nets_out_benchmark_move():
    prices = FakePriceSeriesProvider({
        ("NVDA", "2024-01-01"): 100.0, ("NVDA", "2024-01-21"): 94.0,   # -6%
        ("SPY", "2024-01-01"): 400.0, ("SPY", "2024-01-21"): 404.0,    # +1%
    })
    result = compute_abnormal_return(prices, "NVDA", "SPY", datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    assert result.status == "OK"
    assert result.entity_return == pytest.approx(-0.06)
    assert result.benchmark_return == pytest.approx(0.01)
    assert result.abnormal_return == pytest.approx(-0.07)


def test_missing_price_data_returns_insufficient_data_not_zero():
    prices = FakePriceSeriesProvider({("NVDA", "2024-01-01"): 100.0, ("SPY", "2024-01-01"): 400.0,
                                       ("SPY", "2024-01-21"): 404.0})  # NVDA end price missing
    result = compute_abnormal_return(prices, "NVDA", "SPY", datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.abnormal_return is None


def test_flat_entity_and_benchmark_gives_zero_abnormal_return():
    prices = FakePriceSeriesProvider({("NVDA", "2024-01-01"): 100.0, ("NVDA", "2024-01-21"): 100.0,
                                       ("SPY", "2024-01-01"): 400.0, ("SPY", "2024-01-21"): 400.0})
    result = compute_abnormal_return(prices, "NVDA", "SPY", datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    assert result.abnormal_return == 0.0
