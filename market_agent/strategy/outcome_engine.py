"""Strategy-level outcome engine - stage 7 item 3. Computes GENUINE trade/
path metrics from the actual daily OHLC bars between entry and exit, not
approximations from the final closing return.

PREDICTION OUTCOME vs. TRADE OUTCOME, kept explicitly distinct: an event-
level "prediction outcome" (episodic_events.realized_abnormal_return,
computed by outcomes/observe.py) is the entity's abnormal return over the
horizon, independent of any trading decision. A "trade outcome"
(TradeOutcome, below) exists only for an ACTUAL StrategyAgent LONG/SHORT
decision, is signed by that decision's direction (profit-positive
convention, not raw price return), and adds path-dependent facts
(MFE/MAE, whether the path ever breached the invalidation level) a bare
return figure cannot express. The same real event can have a resolved
PREDICTION outcome while having NO trade outcome at all (StrategyAgent
abstained) - these are never conflated.

MFE/MAE FROM REAL DAILY HIGH/LOW, NOT CLOSE-TO-CLOSE APPROXIMATION: for
each bar in the holding period, the FAVORABLE extreme (in the direction of
the trade) and ADVERSE extreme are computed from that bar's own high/low,
then the max/min across the whole path is taken - the same daily-OHLCV
resolution this project's technical-context layer already discloses as
its ceiling (no intraday/tick data anywhere in this system).

EXIT POLICY IS FIXED_HORIZON ONLY, MATCHING decision_process.py'S OWN
DISCLOSED SCOPE: `hit_invalidation`/`invalidation_day` are reported as
DIAGNOSTIC facts (did the path ever breach the theoretical stop level) -
they do NOT currently trigger an early exit. A genuine stop-honoring exit
simulator is a disclosed, not-yet-built extension, not silently assumed.

SHARED METRICS REUSE experiment/portfolio_metrics.py RATHER THAN
RE-DERIVING THEM: Sharpe, max drawdown, turnover, and transaction-cost-
adjusted total return are computed by converting each TradeOutcome to
that module's TradeRecord and calling its already-tested
compute_portfolio_metrics - one implementation of those metrics, not two.
This module adds what portfolio_metrics.py does not: win rate,
expectancy, profit factor, Sortino, MFE/MAE aggregates, exposure, and
hit-rate conditional on confidence bucket.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from market_agent.experiment.portfolio_metrics import TRANSACTION_COST_PER_TRADE, TradeRecord, compute_portfolio_metrics
from market_agent.outcomes.ohlcv import OHLCVProvider


@dataclass
class TradeOutcome:
    entity: str
    action: str                 # "LONG" | "SHORT"
    entry_date: str             # ISO
    entry_price: float
    exit_date: str              # ISO
    exit_price: float
    horizon_days: int
    holding_days: int           # actual calendar days spanned by the real bars found (may be < horizon_days
    #                             if trailing bars are missing near the end of the cached history)
    raw_entity_return: float    # unsigned entity price return over the path (exit_price/entry_price - 1)
    realized_return: float      # signed by `action` - profit-positive convention
    mfe: float                  # maximum favorable excursion, profit-positive convention, from real daily highs/lows
    mae: float                  # maximum (most negative) adverse excursion, same convention
    hit_invalidation: bool      # diagnostic only - see module docstring
    invalidation_day: int | None
    confidence: str | None = None  # carried through from the StrategyDecision that produced this trade, for
    #                                hit-rate-by-confidence aggregation
    regime: str | None = None      # carried through from the entry context, for strategy_diagnostics.py's
    #                                 pre/post regime-stability check
    path_returns: list[float] = field(default_factory=list)  # signed, action-adjusted cumulative return at
    #   each bar from entry (path_returns[0] is day-of-entry, ~0), for holding-period sensitivity
    #   (strategy/strategy_diagnostics.py) - "what would the return have been if exited at day K instead
    #   of the full horizon."

    def to_trade_record(self, predicted_impact: float) -> TradeRecord:
        """predicted_impact should be the ORIGINAL signed statistical
        prediction (decision.predicted_return) - portfolio_metrics.py's
        own _trade_return derives the direction from ITS sign and
        multiplies by `realized_abnormal_return`, so raw_entity_return
        (unsigned) is the correct value to pass there, not the already
        direction-adjusted realized_return."""
        return TradeRecord(entity=self.entity, triggered_at=self.entry_date, horizon_days=self.horizon_days,
                            predicted_impact=predicted_impact, realized_abnormal_return=self.raw_entity_return)


def compute_trade_outcome(ohlcv: OHLCVProvider, entity: str, action: str, entry_date: datetime,
                           horizon_days: int, invalidation_level: float | None,
                           confidence: str | None = None, regime: str | None = None) -> TradeOutcome | None:
    """Walks the REAL daily bar path from `entry_date` to `entry_date +
    horizon_days`, computing realized return, MFE, and MAE from actual
    highs/lows - never from the final close alone. Returns None if fewer
    than 2 real bars exist in that window (nothing to compute)."""
    exit_target = entry_date + timedelta(days=horizon_days)
    bars = ohlcv.bars(entity, exit_target, lookback_days=horizon_days + 10)
    path = sorted((b for b in bars if b.date >= entry_date), key=lambda b: b.date)
    if len(path) < 2:
        return None

    entry_price = path[0].close
    exit_price = path[-1].close
    if entry_price <= 0:
        return None

    direction_sign = 1.0 if action == "LONG" else -1.0
    raw_entity_return = exit_price / entry_price - 1.0
    realized_return = direction_sign * raw_entity_return

    favorable_returns, adverse_returns, path_returns = [], [], []
    hit_invalidation, invalidation_day = False, None
    for i, bar in enumerate(path):
        if action == "LONG":
            favorable = bar.high / entry_price - 1.0
            adverse = bar.low / entry_price - 1.0
            cumulative = bar.close / entry_price - 1.0
        else:  # SHORT
            favorable = 1.0 - bar.low / entry_price
            adverse = 1.0 - bar.high / entry_price
            cumulative = 1.0 - bar.close / entry_price
        favorable_returns.append(favorable)
        adverse_returns.append(adverse)
        path_returns.append(cumulative)
        if invalidation_level is not None and not hit_invalidation and adverse <= invalidation_level:
            hit_invalidation = True
            invalidation_day = i

    mfe = max(favorable_returns)
    mae = min(adverse_returns)
    holding_days = (path[-1].date - path[0].date).days

    return TradeOutcome(entity=entity, action=action, entry_date=path[0].date.isoformat(), entry_price=entry_price,
                         exit_date=path[-1].date.isoformat(), exit_price=exit_price, horizon_days=horizon_days,
                         holding_days=holding_days, raw_entity_return=raw_entity_return, realized_return=realized_return,
                         mfe=mfe, mae=mae, hit_invalidation=hit_invalidation, invalidation_day=invalidation_day,
                         confidence=confidence, regime=regime, path_returns=path_returns)


@dataclass
class StrategyOutcomeReport:
    n_trades: int
    win_rate: float | None
    expectancy: float | None            # mean realized_return per trade, after transaction costs
    profit_factor: float | None         # sum(gains) / abs(sum(losses)), after costs
    sharpe_annualized: float | None
    sortino_annualized: float | None    # like Sharpe, but only downside deviation in the denominator
    max_drawdown: float | None
    turnover_trades_per_year: float | None
    total_return_after_costs: float | None
    exposure: float | None              # fraction of (decisions considered) that resulted in an actual trade
    mean_mfe: float | None
    mean_mae: float | None
    hit_rate_by_confidence: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


def _sortino_annualized(after_cost_returns: list[float], window_days: float) -> float | None:
    if len(after_cost_returns) < 2:
        return None
    mean = sum(after_cost_returns) / len(after_cost_returns)
    downside = [min(r, 0.0) ** 2 for r in after_cost_returns]
    downside_dev = (sum(downside) / len(downside)) ** 0.5
    if downside_dev == 0:
        return None
    trades_per_year = len(after_cost_returns) / (max(window_days, 1.0) / 365.25)
    return (mean / downside_dev) * math.sqrt(trades_per_year)


def compute_strategy_outcome_report(trades: list[TradeOutcome], n_decisions_considered: int,
                                     predicted_impacts: list[float],
                                     transaction_cost: float = TRANSACTION_COST_PER_TRADE) -> StrategyOutcomeReport:
    """`n_decisions_considered` is the total number of StrategyAgent
    decisions evaluated (LONG/SHORT/ABSTAIN combined) - `exposure` is
    len(trades)/n_decisions_considered, i.e. how often the agent actually
    committed to a position rather than abstaining. `predicted_impacts`
    must be the same length as `trades`, in the same order (the signed
    statistical prediction behind each trade, for portfolio_metrics.py's
    TradeRecord conversion)."""
    if len(trades) < 2:
        return StrategyOutcomeReport(
            len(trades), None, None, None, None, None, None, None, None,
            (len(trades) / n_decisions_considered) if n_decisions_considered else None, None, None,
            evidence=[f"Only {len(trades)} trade(s) - too few for strategy-level metrics to be meaningful."])

    after_cost_returns = [t.realized_return - transaction_cost for t in trades]
    wins = [r for r in after_cost_returns if r > 0]
    losses = [r for r in after_cost_returns if r <= 0]
    win_rate = len(wins) / len(after_cost_returns)
    expectancy = sum(after_cost_returns) / len(after_cost_returns)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None

    records = [t.to_trade_record(p) for t, p in zip(trades, predicted_impacts)]
    portfolio = compute_portfolio_metrics(records, transaction_cost=transaction_cost)

    window_days = portfolio.evaluation_window_days or 1.0
    sortino = _sortino_annualized(after_cost_returns, window_days)

    mean_mfe = sum(t.mfe for t in trades) / len(trades)
    mean_mae = sum(t.mae for t in trades) / len(trades)

    hit_rate_by_confidence: dict[str, list[float]] = {}
    for t, r in zip(trades, after_cost_returns):
        hit_rate_by_confidence.setdefault(t.confidence or "UNKNOWN", []).append(r)
    hit_rate_by_confidence_final = {
        label: sum(1 for r in rs if r > 0) / len(rs) for label, rs in hit_rate_by_confidence.items()}

    exposure = (len(trades) / n_decisions_considered) if n_decisions_considered else None

    evidence = [f"N={len(trades)} trades out of {n_decisions_considered} decisions considered "
                f"(exposure={exposure:.1%})." if exposure is not None else f"N={len(trades)} trades.",
                f"Transaction cost assumption: {transaction_cost:.2%} per round-trip (same as "
                "experiment/portfolio_metrics.py)."]

    return StrategyOutcomeReport(
        n_trades=len(trades), win_rate=win_rate, expectancy=expectancy, profit_factor=profit_factor,
        sharpe_annualized=portfolio.sharpe_annualized, sortino_annualized=sortino,
        max_drawdown=portfolio.max_drawdown, turnover_trades_per_year=portfolio.turnover_trades_per_year,
        total_return_after_costs=portfolio.total_return_after_costs, exposure=exposure, mean_mfe=mean_mfe,
        mean_mae=mean_mae, hit_rate_by_confidence=hit_rate_by_confidence_final, evidence=evidence,
    )
