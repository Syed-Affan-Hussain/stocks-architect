"""Anti-overfitting diagnostics - stage 7. Two real, standard, non-
parametric checks applied to an ALREADY-CONFIRMED condition's matching
rows, from the requested list: a permutation test (covers "permutation
test", "placebo entries", and "shuffled labels" - all the same underlying
technique applied here) and a temporal-stability check.

WHY THESE TWO FIRST, AND WHAT'S DISCLOSED AS NOT BUILT: "regime-specific
performance" and part of "sector-specific performance" are already
substantively covered by Level 3's own context-conditioning tests
(learn/hierarchical_research.py); the rest of the requested list -
parameter perturbation, liquidity buckets, capacity sensitivity, a
separate sector-specific breakdown - would need either a real sector-
classification data source (a disclosed gap since events/schema.py's
ContextSnapshot docstring first noted it) or a position-sizing/capacity
model this project has never built (there is no capital-allocation layer
anywhere in this system - see experiment/portfolio_metrics.py's own
"one notional unit per trade" disclosure). Building those would mean
either fabricating data this system doesn't have or inventing a capacity
model with no real basis - both against this project's standing
discipline. Left as a disclosed gap, not implemented.

BOTH DIAGNOSTICS ARE REPORT-ONLY, never wired into apply_test_results -
same discipline as learn/incremental_value.py. A relationship's SHADOW/
ACTIVE status is decided entirely by the existing, unchanged
learn/hypothesis_testing.py pipeline.

PERMUTATION TEST: answers "is the SPECIFIC SUBSET this condition selects
distinguishable from a random subset of the same size, drawn from the
same broader population (same event_type/direction/horizon, regardless of
technical state)?" This catches a real failure mode Holm-Bonferroni
correction does not fully address: with enough candidate technical
states tested, SOME random-looking subset will look "significant" purely
by chance, even after correction. Comparing against actual random
resampling from the SAME real outcome pool - not a parametric assumption
- is a direct, non-parametric check for that.

TEMPORAL STABILITY: answers "does this condition's effect have the SAME
SIGN in the first half and the second half of its own observed history?"
A sign flip (or a collapse to a near-zero magnitude) across time is a
real red flag that the "effect" may be a period-specific artifact (one
regime, one sector rotation) rather than a persistent pattern.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from market_agent.learn.hypothesis_testing import MIN_N

N_PERMUTATIONS = 2000
PERMUTATION_RNG_SEED = 20240601  # fixed, reproducible - not re-randomized per run


@dataclass
class PermutationTestResult:
    n: int
    observed_mean: float | None
    pool_size: int
    n_permutations: int
    permutation_p_value: float | None
    status: str  # "INSUFFICIENT_N" | "LIKELY_OVERFIT" | "SURVIVES_PERMUTATION"
    evidence: list[str] = field(default_factory=list)


@dataclass
class TemporalStabilityResult:
    n: int
    first_half_n: int
    second_half_n: int
    first_half_mean: float | None
    second_half_mean: float | None
    same_sign: bool | None
    status: str  # "INSUFFICIENT_N" | "UNSTABLE_ACROSS_TIME" | "STABLE_ACROSS_TIME"
    evidence: list[str] = field(default_factory=list)


def _full_pool_rows(conn: sqlite3.Connection, event_type: str, direction: str, horizon_days: int,
                     published_before: str) -> list[sqlite3.Row]:
    """Every resolved event matching event_type/direction/horizon,
    REGARDLESS of technical state - the population the permutation test
    draws random same-size subsets from. Deduplicated by real event for
    the same reason every other statistical sample in this project is
    (store/db.py::deduplicate_by_real_event)."""
    from market_agent.store.db import deduplicate_by_real_event
    rows = conn.execute(
        """SELECT * FROM episodic_events
           WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL AND horizon_days = ?
             AND event_type = ? AND direction = ? AND published_at < ?""",
        (horizon_days, event_type, direction, published_before)).fetchall()
    return deduplicate_by_real_event(rows)


def run_permutation_test(matching_rows: list[sqlite3.Row], pool_rows: list[sqlite3.Row],
                          n_permutations: int = N_PERMUTATIONS, seed: int = PERMUTATION_RNG_SEED
                          ) -> PermutationTestResult:
    n = len(matching_rows)
    pool_size = len(pool_rows)
    if n < MIN_N or pool_size < n:
        return PermutationTestResult(
            n, None, pool_size, n_permutations, None, "INSUFFICIENT_N",
            [f"N={n} matching rows against a pool of {pool_size} - need at least MIN_N={MIN_N} matching rows "
             "and a pool at least as large to draw same-size random subsets from."])

    pool_returns = np.array([r["realized_abnormal_return"] for r in pool_rows])
    observed_mean = float(np.mean([r["realized_abnormal_return"] for r in matching_rows]))
    pool_mean = float(np.mean(pool_returns))
    observed_deviation = abs(observed_mean - pool_mean)

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        sample = rng.choice(pool_returns, size=n, replace=False)
        if abs(float(np.mean(sample)) - pool_mean) >= observed_deviation:
            at_least_as_extreme += 1
    permutation_p = at_least_as_extreme / n_permutations

    evidence = [f"N={n} matching rows drawn from a pool of {pool_size} same-event_type/direction/horizon events.",
                f"Observed subset mean {observed_mean:+.2%} vs. pool mean {pool_mean:+.2%} "
                f"(deviation {observed_deviation:.2%}).",
                f"{n_permutations} random same-size subsets drawn from the SAME pool (seed={seed}, fixed and "
                f"reproducible) - {at_least_as_extreme} of them ({permutation_p:.1%}) deviated from the pool "
                "mean at least as much as the observed subset."]

    status = "LIKELY_OVERFIT" if permutation_p >= 0.05 else "SURVIVES_PERMUTATION"
    if status == "LIKELY_OVERFIT":
        evidence.append("A RANDOM subset of the same size looks at least this extreme too often for this "
                         "condition's selection to be doing real work, independent of Holm-Bonferroni correction.")
    return PermutationTestResult(n, observed_mean, pool_size, n_permutations, permutation_p, status, evidence)


def run_temporal_stability_check(matching_rows: list[sqlite3.Row]) -> TemporalStabilityResult:
    ordered = sorted(matching_rows, key=lambda r: r["published_at"])
    n = len(ordered)
    if n < MIN_N:
        return TemporalStabilityResult(n, 0, 0, None, None, None, "INSUFFICIENT_N",
                                        [f"N={n} - below MIN_N={MIN_N}, cannot split into two meaningful halves."])
    mid = n // 2
    first_half, second_half = ordered[:mid], ordered[mid:]
    first_mean = float(np.mean([r["realized_abnormal_return"] for r in first_half]))
    second_mean = float(np.mean([r["realized_abnormal_return"] for r in second_half]))
    same_sign = (first_mean > 0) == (second_mean > 0)

    evidence = [f"First half (N={len(first_half)}, {first_half[0]['published_at']} to "
                f"{first_half[-1]['published_at']}): mean {first_mean:+.2%}.",
                f"Second half (N={len(second_half)}, {second_half[0]['published_at']} to "
                f"{second_half[-1]['published_at']}): mean {second_mean:+.2%}."]
    status = "STABLE_ACROSS_TIME" if same_sign else "UNSTABLE_ACROSS_TIME"
    if not same_sign:
        evidence.append("Sign flipped between the two halves - the 'effect' may be a period-specific artifact "
                         "rather than a persistent pattern.")
    return TemporalStabilityResult(n, len(first_half), len(second_half), first_mean, second_mean, same_sign,
                                    status, evidence)
