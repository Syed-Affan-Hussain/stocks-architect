"""Outcome observation: realized market-adjusted abnormal return at a
horizon - the same event-study convention EXP_010 used in the deleted
project (raw return alone is dominated by market beta; netting out the
benchmark's contemporaneous move isolates something closer to the
event's own effect).

abnormal_return = entity_return_over_horizon - benchmark_return_over_horizon
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta


class PriceSeriesProvider(ABC):
    """Anything that can answer 'what was this entity's close price on
    this date'. Abstracted specifically so tests never touch the network
    (see tests/test_outcomes_observe.py's FakePriceSeriesProvider) and so
    a live implementation (e.g. yfinance-backed) can be swapped in without
    touching the abnormal-return math below."""

    @abstractmethod
    def close_price(self, ticker: str, as_of: datetime) -> float | None:
        """Close price on or immediately before `as_of`. None if unavailable -
        callers must treat this as INSUFFICIENT_DATA, never as a silent zero."""
        raise NotImplementedError


@dataclass
class AbnormalReturnResult:
    status: str  # "OK" | "INSUFFICIENT_DATA"
    abnormal_return: float | None
    entity_return: float | None
    benchmark_return: float | None
    evidence: list[str]


def compute_abnormal_return(prices: PriceSeriesProvider, ticker: str, benchmark_ticker: str,
                             event_date: datetime, horizon_days: int) -> AbnormalReturnResult:
    evidence = []
    start = event_date
    end = event_date + timedelta(days=horizon_days)

    entity_start, entity_end = prices.close_price(ticker, start), prices.close_price(ticker, end)
    bench_start, bench_end = prices.close_price(benchmark_ticker, start), prices.close_price(benchmark_ticker, end)

    if None in (entity_start, entity_end, bench_start, bench_end):
        evidence.append(f"Missing price data for {ticker} or {benchmark_ticker} at start/end of the "
                         f"{horizon_days}-day window - cannot compute abnormal return.")
        return AbnormalReturnResult("INSUFFICIENT_DATA", None, None, None, evidence)
    if entity_start <= 0 or bench_start <= 0:
        evidence.append("Non-positive start price - cannot compute a return.")
        return AbnormalReturnResult("INSUFFICIENT_DATA", None, None, None, evidence)

    entity_return = entity_end / entity_start - 1.0
    benchmark_return = bench_end / bench_start - 1.0
    abnormal_return = entity_return - benchmark_return
    evidence.append(f"{ticker}: {entity_return:+.2%} over {horizon_days}d; {benchmark_ticker}: "
                     f"{benchmark_return:+.2%}; abnormal (market-adjusted): {abnormal_return:+.2%}")
    return AbnormalReturnResult("OK", abnormal_return, entity_return, benchmark_return, evidence)
