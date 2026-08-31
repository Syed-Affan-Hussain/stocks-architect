"""Real Yahoo Finance price provider - implements outcomes.observe's
PriceSeriesProvider interface for real historical backtesting, and (stage
6) outcomes.ohlcv's OHLCVProvider interface for the technical-concept
layer, which needs real highs/lows/volume, not just closes.

Fetches each ticker's full daily history ONCE (a wide window covering the
whole experiment period, not a per-prediction call) and caches it to
parquet, then answers queries purely from that local cache - the same
"ingest real history once, replay many times locally" pattern as
sources/edgar_guidance.py, and for the same reason: point-in-time replay
should query a fixed, already-fetched historical dataset, not make a
fresh live call whose result could in principle differ depending on when
the backtest happens to run.

CACHE FORMAT MIGRATION (stage 6): pre-stage-6 parquet files hold only a
`close` column - the full OHLCV frame is now cached instead. An old-format
file is detected (missing the open/high/low/volume columns) and
transparently re-fetched/re-cached on first read, exactly the same
disclosed migrate-on-read idiom store/schema.py's `_migrate()` uses for
the SQLite schema (PRAGMA table_info check, then ALTER) - this does not
lose or fabricate any history, it just re-pulls the same real Yahoo
history with more columns kept this time.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class YahooPriceSeriesProvider(PriceSeriesProvider, OHLCVProvider):
    def __init__(self, cache_dir: str | Path = "data_cache/prices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, pd.DataFrame] = {}

    def _fetch_and_cache(self, ticker: str, start: str, end: str | None, cache_path: Path) -> pd.DataFrame:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist is None or len(hist) == 0:
            frame = pd.DataFrame(columns=OHLCV_COLUMNS)
            frame.index = pd.to_datetime(frame.index, utc=True)
        else:
            frame = hist.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                          "Close": "close", "Volume": "volume"})[OHLCV_COLUMNS]
            frame.index = pd.to_datetime(frame.index, utc=True)
        frame.to_parquet(cache_path)
        return frame

    def _load_frame(self, ticker: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
        if ticker in self._loaded:
            return self._loaded[ticker]
        cache_path = self.cache_dir / f"{ticker}.parquet"
        if cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if not set(OHLCV_COLUMNS).issubset(frame.columns):
                frame = self._fetch_and_cache(ticker, start, end, cache_path)  # old close-only cache - migrate
            else:
                frame.index = pd.to_datetime(frame.index, utc=True)
        else:
            frame = self._fetch_and_cache(ticker, start, end, cache_path)
        self._loaded[ticker] = frame
        return frame

    def _load(self, ticker: str, start: str = "2015-01-01", end: str | None = None) -> pd.Series:
        """Kept for backward compatibility with existing call sites
        (scripts pre-warming the cache via `prices._load(ticker)`) - now
        just the close column of the full OHLCV frame."""
        return self._load_frame(ticker, start, end)["close"]

    def close_price(self, ticker: str, as_of: datetime) -> float | None:
        try:
            series = self._load(ticker)
            if series.empty:
                return None
            as_of_ts = pd.Timestamp(as_of)
            as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
            eligible = series[series.index <= as_of_ts]
            if eligible.empty:
                return None
            return float(eligible.iloc[-1])
        except Exception:  # noqa: BLE001
            # A single ticker's malformed/corrupted cached series (e.g. a degenerate index on a
            # near-empty/delisted history - observed live for a handful of real tickers) must
            # degrade to "price unavailable" for that one lookup, never crash a whole experiment
            # run. compute_abnormal_return already treats None as INSUFFICIENT_DATA, not a
            # fabricated zero - this is the same fail-closed convention, just also catching an
            # unexpected exception rather than only an expected missing-key case.
            return None

    def bars(self, ticker: str, as_of: datetime, lookback_days: int) -> list[Bar]:
        try:
            frame = self._load_frame(ticker)
            if frame.empty:
                return []
            as_of_ts = pd.Timestamp(as_of)
            as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
            start_ts = as_of_ts - pd.Timedelta(days=lookback_days)
            window = frame[(frame.index <= as_of_ts) & (frame.index >= start_ts)]
            if window.empty:
                return []
            return [Bar(date=idx.to_pydatetime(), open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                        volume=(float(row["volume"]) if pd.notna(row["volume"]) else None))
                    for idx, row in window.iterrows()]
        except Exception:  # noqa: BLE001
            # Same fail-closed convention as close_price() above - a malformed cache for one
            # ticker degrades to "no bars available", never crashes the caller.
            return []
