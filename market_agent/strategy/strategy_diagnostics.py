"""Strategy-level anti-overfitting diagnostics - stage 7 item 5. Extends
learn/overfitting_diagnostics.py's concept-level checks (permutation test,
temporal stability) to the TRADE level, and adds the strategy-specific
checks that module doesn't cover: transaction-cost sensitivity,
entry-threshold sensitivity, holding-period sensitivity, pre/post-regime
stability, placebo-strategy comparison, bootstrap confidence intervals,
and TRAIN->VALIDATE->SHADOW->TEST performance-degradation reporting.

ALL DIAGNOSTIC, NONE A PROMOTION GATE: exactly the same discipline as
learn/incremental_value.py and learn/overfitting_diagnostics.py - nothing
here writes to validated_relationships or influences apply_test_results.
These functions exist to be READ, by a human or by the final report
(item 9), never to silently retune anything - see
strategy/test_isolation.py for the explicit mechanism preventing a
parameter from being selected AFTER any of these are computed on TEST
data.

"Do not simply report statistical significance. Report whether the
economic effect survives reasonable perturbations." - every function here
reports a RANGE or a comparison, never a single pass/fail number standing
in for robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from market_agent.strategy.outcome_engine import StrategyOutcomeReport, TradeOutcome, compute_strategy_outcome_report

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20240601  # fixed, reproducible


# --- bootstrap confidence intervals ---

def bootstrap_confidence_interval(values: list[float], seed: int = BOOTSTRAP_SEED,
                                   n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float] | None:
    """95% percentile bootstrap CI on the MEAN of `values`. Returns None
    if fewer than 2 values (nothing to resample meaningfully)."""
    if len(values) < 2:
        return None
    arr = np.array(values)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = arr[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- transaction-cost sensitivity ---

def transaction_cost_sensitivity(trades: list[TradeOutcome], predicted_impacts: list[float],
                                  cost_grid: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005)
                                  ) -> dict[float, StrategyOutcomeReport]:
    """Re-scores the SAME trade list under a fixed, disclosed grid of
    transaction-cost assumptions - shows whether the strategy's apparent
    edge survives realistic cost uncertainty, not just the single
    TRANSACTION_COST_PER_TRADE point estimate."""
    return {cost: compute_strategy_outcome_report(trades, len(trades), predicted_impacts, transaction_cost=cost)
            for cost in cost_grid}


# --- entry-threshold (cost-margin) sensitivity ---

def threshold_sensitivity(decision_fn, cost_margin_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
                           ) -> dict[float, dict[str, int]]:
    """`decision_fn(cost_margin_multiple) -> list[StrategyDecision]` -
    the caller re-runs StrategyAgent with each threshold and passes back
    the resulting decisions (kept as a callback rather than this module
    owning StrategyAgent construction, so the SAME real decision
    processes/contexts are reused at every threshold, not resampled).
    Reports the action-count breakdown at each threshold - how many
    LONG/SHORT/ABSTAIN decisions result, showing how sensitive the
    strategy's activity level is to this one gate."""
    results = {}
    for multiple in cost_margin_grid:
        decisions = decision_fn(multiple)
        counts = {"LONG": 0, "SHORT": 0, "ABSTAIN": 0}
        for d in decisions:
            counts[d.action] = counts.get(d.action, 0) + 1
        results[multiple] = counts
    return results


# --- holding-period sensitivity ---

def holding_period_sensitivity(trades: list[TradeOutcome],
                                day_grid: tuple[int, ...] = (1, 5, 10, 20, 40, 60)) -> dict[int, float | None]:
    """For each day K in `day_grid`, the mean realized return HAD every
    trade been exited at day K instead of its actual fixed-horizon exit -
    using the REAL per-bar path already recorded in
    TradeOutcome.path_returns (outcomes/outcome_engine.py), never a
    reconstruction. None for a K beyond what any trade's path reaches."""
    results: dict[int, float | None] = {}
    for day in day_grid:
        returns_at_day = [t.path_returns[day] for t in trades if len(t.path_returns) > day]
        results[day] = (sum(returns_at_day) / len(returns_at_day)) if returns_at_day else None
    return results


# --- pre/post regime stability ---

def regime_stability(trades: list[TradeOutcome], predicted_impacts: list[float]) -> dict[str, StrategyOutcomeReport]:
    """Splits trades by their OWN entry-time regime (TradeOutcome.regime)
    and reports strategy-level metrics separately per regime - a real
    edge should not depend entirely on one regime bucket."""
    by_regime: dict[str, list[tuple[TradeOutcome, float]]] = {}
    for t, p in zip(trades, predicted_impacts):
        by_regime.setdefault(t.regime or "UNKNOWN", []).append((t, p))
    return {regime: compute_strategy_outcome_report([t for t, _ in pairs], len(pairs), [p for _, p in pairs])
            for regime, pairs in by_regime.items()}


# --- TRAIN -> VALIDATE -> SHADOW -> TEST performance degradation ---

def segment_degradation(trades_by_segment: dict[str, list[TradeOutcome]],
                         predicted_impacts_by_segment: dict[str, list[float]]) -> dict[str, StrategyOutcomeReport]:
    """One StrategyOutcomeReport per named segment (e.g. "VALIDATE",
    "TEST") - the caller compares them (e.g. TEST expectancy vs. VALIDATE
    expectancy) to see whether performance degrades moving from
    training-adjacent segments toward the untouched final evaluation, a
    classic overfitting signature. This function computes each segment's
    numbers; it does not itself declare "degraded" or "fine" - that
    judgment belongs in the final report (item 9), not buried in a
    boolean here."""
    return {
        segment: compute_strategy_outcome_report(trades, len(trades), predicted_impacts_by_segment[segment])
        for segment, trades in trades_by_segment.items()
    }


# --- placebo strategy test ---

def run_placebo_strategy_test(ohlcv, entities: list[str], window_start: datetime, window_end: datetime,
                               n_placebo_trades: int, horizon_days: int, action_pool: tuple[str, ...] = ("LONG", "SHORT"),
                               seed: int = BOOTSTRAP_SEED) -> StrategyOutcomeReport:
    """Real (not fabricated) trade outcomes computed via
    outcome_engine.compute_trade_outcome, but for RANDOM entities, random
    entry dates within [window_start, window_end), and a random LONG/SHORT
    action - i.e. entries with NO connection to any validated
    setup/regime condition. If the real strategy's performance is not
    clearly better than this placebo's, the real strategy provides no
    demonstrated edge over trading blind."""
    from market_agent.strategy.outcome_engine import compute_trade_outcome

    rng = np.random.default_rng(seed)
    window_days = max((window_end - window_start).days, 1)
    trades: list[TradeOutcome] = []
    predicted_impacts: list[float] = []
    attempts = 0
    max_attempts = n_placebo_trades * 20  # bounded - real cached data may not cover every random draw
    while len(trades) < n_placebo_trades and attempts < max_attempts:
        attempts += 1
        entity = entities[rng.integers(0, len(entities))]
        offset_days = int(rng.integers(0, window_days))
        entry_date = window_start + timedelta(days=offset_days)
        action = action_pool[rng.integers(0, len(action_pool))]
        outcome = compute_trade_outcome(ohlcv, entity, action, entry_date, horizon_days, invalidation_level=None)
        if outcome is not None:
            trades.append(outcome)
            predicted_impacts.append(0.0)  # a placebo has no real statistical prediction behind it

    return compute_strategy_outcome_report(trades, len(trades), predicted_impacts)


# --- strategy-level permutation test (observed trades vs. a placebo/random pool) ---

@dataclass
class StrategyPermutationResult:
    n_observed: int
    n_pool: int
    observed_mean_return: float | None
    pool_mean_return: float | None
    n_permutations: int
    permutation_p_value: float | None
    status: str  # "INSUFFICIENT_N" | "LIKELY_OVERFIT" | "SURVIVES_PERMUTATION"
    evidence: list[str] = field(default_factory=list)


def run_strategy_permutation_test(observed_trades: list[TradeOutcome], pool_trades: list[TradeOutcome],
                                   n_permutations: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                                   min_n: int = 15) -> StrategyPermutationResult:
    """Is the OBSERVED trade set's mean realized return distinguishable
    from `n_observed`-sized random draws from `pool_trades` (typically the
    placebo/random-entry pool from run_placebo_strategy_test)? Same
    logic as learn/overfitting_diagnostics.py's concept-level permutation
    test, applied to trade-level realized returns instead of event-level
    abnormal returns."""
    n_observed, n_pool = len(observed_trades), len(pool_trades)
    if n_observed < min_n or n_pool < n_observed:
        return StrategyPermutationResult(
            n_observed, n_pool, None, None, n_permutations, None, "INSUFFICIENT_N",
            [f"N={n_observed} observed trades vs. a pool of {n_pool} - need at least {min_n} observed trades "
             "and a pool at least as large."])

    pool_returns = np.array([t.realized_return for t in pool_trades])
    observed_mean = float(np.mean([t.realized_return for t in observed_trades]))
    pool_mean = float(np.mean(pool_returns))
    observed_deviation = abs(observed_mean - pool_mean)

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        sample = rng.choice(pool_returns, size=n_observed, replace=False)
        if abs(float(np.mean(sample)) - pool_mean) >= observed_deviation:
            at_least_as_extreme += 1
    p_value = at_least_as_extreme / n_permutations

    evidence = [f"N={n_observed} observed trades vs. a pool of {n_pool} placebo/random trades.",
                f"Observed mean return {observed_mean:+.2%} vs. pool mean {pool_mean:+.2%} "
                f"(deviation {observed_deviation:.2%}).",
                f"{n_permutations} random same-size draws from the pool (seed={seed}) - {at_least_as_extreme} "
                f"of them ({p_value:.1%}) deviated at least as much."]
    status = "LIKELY_OVERFIT" if p_value >= 0.05 else "SURVIVES_PERMUTATION"
    return StrategyPermutationResult(n_observed, n_pool, observed_mean, pool_mean, n_permutations, p_value,
                                      status, evidence)


# --- strategy-level temporal stability ---

@dataclass
class StrategyTemporalStabilityResult:
    n: int
    first_half_mean: float | None
    second_half_mean: float | None
    same_sign: bool | None
    status: str  # "INSUFFICIENT_N" | "UNSTABLE_ACROSS_TIME" | "STABLE_ACROSS_TIME"
    evidence: list[str] = field(default_factory=list)


def run_strategy_temporal_stability(trades: list[TradeOutcome], min_n: int = 15) -> StrategyTemporalStabilityResult:
    ordered = sorted(trades, key=lambda t: t.entry_date)
    n = len(ordered)
    if n < min_n:
        return StrategyTemporalStabilityResult(n, None, None, None, "INSUFFICIENT_N",
                                                [f"N={n} - below the minimum of {min_n}."])
    mid = n // 2
    first_mean = sum(t.realized_return for t in ordered[:mid]) / mid
    second_mean = sum(t.realized_return for t in ordered[mid:]) / (n - mid)
    same_sign = (first_mean > 0) == (second_mean > 0)
    status = "STABLE_ACROSS_TIME" if same_sign else "UNSTABLE_ACROSS_TIME"
    evidence = [f"First half (N={mid}): mean {first_mean:+.2%}.", f"Second half (N={n - mid}): mean {second_mean:+.2%}."]
    if not same_sign:
        evidence.append("Sign flipped between halves - a real red flag for a period-specific artifact.")
    return StrategyTemporalStabilityResult(n, first_mean, second_mean, same_sign, status, evidence)
