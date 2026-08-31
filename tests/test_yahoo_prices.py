"""Network-free tests for sources/yahoo_prices.py's OHLCV cache and the
stage-6 cache-format migration. Live Yahoo Finance calls are exercised
only by the real-data experiment scripts, same testing boundary as
sources/edgar_guidance.py/edgar_dividend.py.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from market_agent.sources.yahoo_prices import OHLCV_COLUMNS, YahooPriceSeriesProvider

AS_OF = datetime(2024, 3, 15, tzinfo=timezone.utc)


def _write_ohlcv_cache(cache_dir, ticker, rows):
    """rows: list of (date_str, open, high, low, close, volume)."""
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    frame = pd.DataFrame({
        "open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
        "close": [r[4] for r in rows], "volume": [r[5] for r in rows],
    }, index=idx)
    (cache_dir / f"{ticker}.parquet").parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_dir / f"{ticker}.parquet")


def test_bars_reads_full_ohlcv_from_cache_without_network(tmp_path):
    _write_ohlcv_cache(tmp_path, "ACME", [
        ("2024-03-10", 100.0, 105.0, 99.0, 103.0, 1_000_000),
        ("2024-03-11", 103.0, 108.0, 102.0, 107.0, 1_200_000),
        ("2024-03-12", 107.0, 110.0, 106.0, 109.0, 900_000),
    ])
    provider = YahooPriceSeriesProvider(cache_dir=tmp_path)
    bars = provider.bars("ACME", datetime(2024, 3, 12, tzinfo=timezone.utc), lookback_days=10)
    assert len(bars) == 3
    assert bars[0].open == 100.0 and bars[0].high == 105.0 and bars[0].low == 99.0
    assert bars[-1].close == 109.0
    assert bars[-1].volume == 900_000


def test_bars_is_point_in_time_and_respects_lookback(tmp_path):
    _write_ohlcv_cache(tmp_path, "ACME", [
        ("2024-01-01", 100.0, 101.0, 99.0, 100.5, 500_000),
        ("2024-03-10", 100.0, 105.0, 99.0, 103.0, 1_000_000),
        ("2024-03-15", 103.0, 108.0, 102.0, 107.0, 1_200_000),  # after as_of - must never appear
    ])
    provider = YahooPriceSeriesProvider(cache_dir=tmp_path)
    bars = provider.bars("ACME", datetime(2024, 3, 12, tzinfo=timezone.utc), lookback_days=5)
    dates = [b.date.date().isoformat() for b in bars]
    assert "2024-03-15" not in dates  # future bar excluded
    assert "2024-01-01" not in dates  # outside the 5-day lookback window
    assert "2024-03-10" in dates


def test_bars_returns_empty_list_for_missing_ticker(tmp_path):
    provider = YahooPriceSeriesProvider(cache_dir=tmp_path)

    class _EmptyHist:
        def history(self, **kwargs):
            return pd.DataFrame()

    import market_agent.sources.yahoo_prices as mod
    orig_ticker = mod.yf.Ticker
    mod.yf.Ticker = lambda t: _EmptyHist()
    try:
        bars = provider.bars("NOSUCHTICKER", AS_OF, lookback_days=10)
        assert bars == []
    finally:
        mod.yf.Ticker = orig_ticker


def test_close_price_still_works_against_the_new_ohlcv_cache_format(tmp_path):
    _write_ohlcv_cache(tmp_path, "ACME", [
        ("2024-03-10", 100.0, 105.0, 99.0, 103.0, 1_000_000),
        ("2024-03-12", 107.0, 110.0, 106.0, 109.0, 900_000),
    ])
    provider = YahooPriceSeriesProvider(cache_dir=tmp_path)
    assert provider.close_price("ACME", datetime(2024, 3, 12, tzinfo=timezone.utc)) == 109.0


def test_old_close_only_cache_is_migrated_on_read(tmp_path, monkeypatch):
    old_cache = pd.DataFrame({"close": [100.0, 101.0]},
                              index=pd.to_datetime(["2024-03-10", "2024-03-11"], utc=True))
    (tmp_path / "ACME.parquet").parent.mkdir(parents=True, exist_ok=True)
    old_cache.to_parquet(tmp_path / "ACME.parquet")
    assert set(OHLCV_COLUMNS) - set(pd.read_parquet(tmp_path / "ACME.parquet").columns) == \
        {"open", "high", "low", "volume"}

    class _FakeHist:
        def history(self, **kwargs):
            return pd.DataFrame({
                "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0],
                "Close": [101.5, 102.5], "Volume": [1_000_000, 1_100_000],
            }, index=pd.to_datetime(["2024-03-10", "2024-03-11"]))

    import market_agent.sources.yahoo_prices as mod
    monkeypatch.setattr(mod.yf, "Ticker", lambda t: _FakeHist())

    provider = YahooPriceSeriesProvider(cache_dir=tmp_path)
    bars = provider.bars("ACME", datetime(2024, 3, 11, tzinfo=timezone.utc), lookback_days=10)
    assert len(bars) == 2
    assert bars[0].high == 102.0  # came from the re-fetched frame, not the old close-only cache

    migrated = pd.read_parquet(tmp_path / "ACME.parquet")
    assert set(OHLCV_COLUMNS).issubset(migrated.columns)
