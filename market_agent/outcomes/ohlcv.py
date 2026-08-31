"""Point-in-time OHLCV access - stage 6's foundation for the technical
concept layer (concepts/technical_context.py needs real highs/lows/volume,
not just closes).

A SEPARATE, ADDITIVE interface from PriceSeriesProvider (outcomes/observe.py
already has a "close-price point lookup" contract), not a replacement or
extension of it - every existing close-price-only test double (this
project's various SyntheticPrices/FlatPrices fixtures across the test
suite, and any future one) keeps working completely unchanged. Only a
provider that actually HAS OHLCV data (YahooPriceSeriesProvider) needs to
implement OHLCVProvider too; a provider that only ever had a close-price
series (a hand-built synthetic fixture, for instance) simply doesn't.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None  # None, never a fabricated 0, when the source genuinely has no volume figure


class OHLCVProvider(ABC):
    @abstractmethod
    def bars(self, ticker: str, as_of: datetime, lookback_days: int) -> list[Bar]:
        """Bars strictly on or before `as_of`, oldest first, going back at
        most `lookback_days` calendar days (fewer if history doesn't reach
        that far back, or is missing/delisted - an empty list, never a
        fabricated bar). Point-in-time by construction: a conforming
        implementation must never return a bar dated after `as_of`."""
        raise NotImplementedError
