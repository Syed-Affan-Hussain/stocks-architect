"""Trading-style backtest metrics - stage 6 item: Sharpe, max drawdown,
turnover, transaction-cost-adjusted return, expected value - to sit
alongside experiment/metrics.py's prediction-accuracy metrics (direction
accuracy, MAE, RMSE, Brier, calibration) in the four-way comparison.

A DISCLOSED, SIMPLE TRADING-RULE ASSUMPTION MAKES THIS COMPUTABLE AT ALL:
this system predicts an abnormal return for one EVENT at one HORIZON, not
a continuously-held portfolio position - there is no position-sizing,
leverage, or capital-allocation model anywhere in this project. To turn a
stream of event-level predictions into portfolio-style metrics at all,
each (event, horizon) with a non-null prediction and a resolved outcome is
treated as exactly ONE NOTIONAL UNIT round-trip trade: long if
predicted_impact > 0, short if < 0, held for the horizon, closed at
realized_abnormal_return. This is a real, standard event-study backtest
convention - not a fabricated portfolio simulation - but it has real,
disclosed limitations:

  - Trades are NOT capital-weighted or position-sized; every trade counts
    equally regardless of predicted_confidence or novelty_score.
  - Overlapping holding periods are NOT modeled - two trades 5 calendar
    days apart both count fully in the equity curve even though a real
    trader couldn't hold both without more capital. This is why turnover
    here is reported as trades-per-year (a rate), never a dollar-turnover
    figure this system has no capital base to compute honestly.
  - The equity curve is a SIMPLE (non-compounded) running sum of trade
    returns, not geometric compounding - standard for event-study
    abnormal-return aggregation, but explicitly not the same as a real
    portfolio NAV curve.
  - Sharpe's annualization assumes i.i.d. trade returns, which overlapping
    holding periods and shared market-wide event days almost certainly
    violate - reported anyway (it's the standard formula), but this
    caveat is not hidden.

TRANSACTION_COST_PER_TRADE is a fixed, disclosed round-trip cost assumption
(10 basis points - a standard, unremarkable order-of-magnitude figure for
a liquid US equity), not fit to any backtest result and not a claim about
any specific broker's real costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

TRANSACTION_COST_PER_TRADE = 0.001  # 10bps round-trip, disclosed fixed assumption - see module docstring
TRADING_DAYS_PER_YEAR = 252


@dataclass
class TradeRecord:
    entity: str
    triggered_at: str  # ISO timestamp - chronological ordering key
    horizon_days: int
    predicted_impact: float
    realized_abnormal_return: float


@dataclass
class PortfolioMetricsReport:
    n_trades: int
    expected_value: float | None                # mean trade return, before transaction costs
    expected_value_after_costs: float | None     # mean trade return, after TRANSACTION_COST_PER_TRADE
    total_return_after_costs: float | None       # simple (non-compounded) sum of after-cost trade returns
    sharpe_per_trade: float | None                # mean/stdev of the after-cost trade-return distribution
    sharpe_annualized: float | None               # sharpe_per_trade * sqrt(trades_per_year) - see caveat above
    max_drawdown: float | None                    # most negative peak-to-trough decline of the equity curve
    turnover_trades_per_year: float | None
    evaluation_window_days: float | None
    sortino_per_trade: float | None = None        # mean/DOWNSIDE-deviation only (Sortino & Price, 1994) -
    #    same after-cost trade-return distribution as Sharpe above, but the denominator only counts
    #    below-target (here: below 0) returns, since Sortino's whole point is that upside variance
    #    shouldn't penalize a strategy the way Sharpe's total-variance denominator does. None if fewer
    #    than 2 losing trades exist (no downside deviation to compute - not 0, which would wrongly
    #    imply "flawless downside").
    sortino_annualized: float | None = None       # sortino_per_trade * sqrt(trades_per_year), same
    #    annualization convention (and same i.i.d. caveat) as sharpe_annualized above
    evidence: list[str] = field(default_factory=list)


def _trade_return(predicted_impact: float, realized_abnormal_return: float) -> float:
    direction = 1.0 if predicted_impact > 0 else (-1.0 if predicted_impact < 0 else 0.0)
    return direction * realized_abnormal_return


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _downside_deviation(values: list[float], target: float = 0.0) -> float | None:
    """Sortino & Price (1994): stdev computed ONLY over returns below
    `target` (0.0 here - the same "no change" reference point Sharpe
    implicitly uses via the raw return itself). None if fewer than 2
    below-target returns exist - not 0.0, which would misrepresent "too
    little data" as "measured zero downside risk"."""
    downside = [min(v - target, 0.0) for v in values if v < target]
    if len(downside) < 2:
        return None
    mean_sq = sum(d ** 2 for d in downside) / len(downside)
    return mean_sq ** 0.5


def compute_portfolio_metrics(trades: list[TradeRecord],
                               transaction_cost: float = TRANSACTION_COST_PER_TRADE) -> PortfolioMetricsReport:
    if len(trades) < 2:
        return PortfolioMetricsReport(
            n_trades=len(trades), expected_value=None, expected_value_after_costs=None,
            total_return_after_costs=None, sharpe_per_trade=None, sharpe_annualized=None,
            max_drawdown=None, turnover_trades_per_year=None, evaluation_window_days=None,
            evidence=[f"Only {len(trades)} trade(s) - too few for any portfolio-style metric to be meaningful."])

    ordered = sorted(trades, key=lambda t: t.triggered_at)
    raw_returns = [_trade_return(t.predicted_impact, t.realized_abnormal_return) for t in ordered]
    after_cost_returns = [r - transaction_cost for r in raw_returns]

    expected_value = sum(raw_returns) / len(raw_returns)
    expected_value_after_costs = sum(after_cost_returns) / len(after_cost_returns)
    total_return_after_costs = sum(after_cost_returns)

    sd = _stdev(after_cost_returns)
    sharpe_per_trade = (expected_value_after_costs / sd) if sd and sd > 0 else None

    dd = _downside_deviation(after_cost_returns)
    sortino_per_trade = (expected_value_after_costs / dd) if dd and dd > 0 else None

    from datetime import datetime
    start = datetime.fromisoformat(ordered[0].triggered_at)
    end = datetime.fromisoformat(ordered[-1].triggered_at)
    window_days = max((end - start).total_seconds() / 86400.0, 1.0)
    trades_per_year = len(ordered) / (window_days / 365.25)
    sharpe_annualized = (sharpe_per_trade * math.sqrt(trades_per_year)) if sharpe_per_trade is not None else None
    sortino_annualized = (sortino_per_trade * math.sqrt(trades_per_year)) if sortino_per_trade is not None else None

    equity = 0.0
    running_max = 0.0
    max_dd = 0.0
    for r in after_cost_returns:
        equity += r
        running_max = max(running_max, equity)
        max_dd = min(max_dd, equity - running_max)

    evidence = [f"N={len(ordered)} trades, one notional unit per (event, horizon) - see module docstring.",
                f"Evaluation window: {window_days:.0f} days ({ordered[0].triggered_at} to {ordered[-1].triggered_at}).",
                f"Transaction cost assumption: {transaction_cost:.2%} per round-trip trade (fixed, disclosed)."]

    return PortfolioMetricsReport(
        n_trades=len(ordered), expected_value=expected_value, expected_value_after_costs=expected_value_after_costs,
        total_return_after_costs=total_return_after_costs, sharpe_per_trade=sharpe_per_trade,
        sharpe_annualized=sharpe_annualized, max_drawdown=max_dd, turnover_trades_per_year=trades_per_year,
        evaluation_window_days=window_days, sortino_per_trade=sortino_per_trade,
        sortino_annualized=sortino_annualized, evidence=evidence,
    )
