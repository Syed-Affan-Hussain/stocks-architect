"""Incremental-value testing - stage 7: "a technical methodology should
not be considered useful merely because it predicts returns... test
E[return | methodology] - E[return | existing model/context]."

WHY THIS IS A SEPARATE, ADDITIONAL CHECK, NOT A REPLACEMENT FOR THE
EXISTING SIGNIFICANCE TEST: learn/hypothesis_testing.py's
_run_significance_test compares a condition's mean realized return
against `unconditional_baseline` - a single, FIXED, burn-in-derived
scalar per horizon. That comparison answers "does this condition differ
from a crude population-average magnitude" - a real question, but NOT
the same question as "does this condition tell the EXISTING ADAPTIVE
MODEL something it doesn't already know." A condition can clear the
first test while adding nothing over CURRENT_ADAPTIVE's own live
prediction for the very same cases (which may already be conditioning on
regime/prior_return_bucket/vol_bucket and beating the crude scalar
itself). This module tests the second, harder question directly, by
comparing against CURRENT_ADAPTIVE's OWN LOGGED prediction for each
matching real event - not a fixed number.

THIS DOES NOT CHANGE ANY GOVERNANCE DECISION: it is never wired into
apply_test_results/promote_from_shadow - a relationship's SHADOW/ACTIVE
status is decided ENTIRELY by the existing, unchanged
learn/hypothesis_testing.py pipeline (preserving the existing governance
architecture exactly, per this stage's explicit instruction). This
module's result is an ADDITIONAL, separately reported diagnostic -
learn/hierarchical_research.py attaches one alongside each Level 2/3
result so a reader can see, honestly, which confirmed setups also clear
this harder bar and which merely differ from the crude baseline scalar.

Uses the SAME MIN_N/ALPHA/MIN_ECONOMIC_EFFECT constants as
learn/hypothesis_testing.py - not a separately tunable set of gates.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from scipy import stats

from market_agent.learn.hypothesis_testing import ALPHA, MIN_ECONOMIC_EFFECT, MIN_N


@dataclass
class IncrementalValueResult:
    condition_label: str
    n: int
    mean_incremental_diff: float | None    # mean(realized - CURRENT_ADAPTIVE's own prediction)
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    status: str  # "INSUFFICIENT_N" | "NO_INCREMENTAL_VALUE" | "INCREMENTAL_VALUE_CONFIRMED"
    evidence: list[str] = field(default_factory=list)


def _current_adaptive_predicted_impact(conn: sqlite3.Connection, entity: str, published_at: str, horizon_days: int,
                                        event_type: str, direction: str) -> float | None:
    row = conn.execute(
        """SELECT predicted_impact FROM episodic_events
           WHERE entity = ? AND published_at = ? AND horizon_days = ? AND event_type = ? AND direction = ?
             AND model_version LIKE 'CURRENT_ADAPTIVE%' AND predicted_impact IS NOT NULL
           LIMIT 1""",
        (entity, published_at, horizon_days, event_type, direction)).fetchone()
    return row["predicted_impact"] if row else None


def test_incremental_value(conn: sqlite3.Connection, condition: dict, horizon_days: int,
                            matching_rows: list[sqlite3.Row]) -> IncrementalValueResult:
    """`matching_rows` should already be deduplicated by real event (see
    store/db.py::deduplicate_by_real_event) - this function looks up
    CURRENT_ADAPTIVE's own prediction for each one independently, so it
    does not matter which agent's row survived that deduplication."""
    condition_label = " AND ".join(f"{k}={v!r}" for k, v in condition.items())
    diffs = []
    for row in matching_rows:
        existing_pred = _current_adaptive_predicted_impact(conn, row["entity"], row["published_at"], horizon_days,
                                                             condition["event_type"], condition["direction"])
        if existing_pred is None:
            continue
        diffs.append(row["realized_abnormal_return"] - existing_pred)

    n = len(diffs)
    if n < MIN_N:
        return IncrementalValueResult(
            condition_label, n, None, None, None, None, "INSUFFICIENT_N",
            [f"Only {n} matching case(s) had an existing CURRENT_ADAPTIVE prediction logged for the same real "
             f"event - below MIN_N={MIN_N}. Cannot assess incremental value either way."])

    t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0)
    mean_diff = sum(diffs) / n
    sem = (stats.tstd(diffs) / (n ** 0.5)) if n > 1 else 0.0
    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci_low, ci_high = mean_diff - t_crit * sem, mean_diff + t_crit * sem

    evidence = [f"N={n} cases with an existing CURRENT_ADAPTIVE prediction for the same real event.",
                f"Mean (realized - CURRENT_ADAPTIVE's own prediction): {mean_diff:+.2%} "
                f"(95% CI [{ci_low:+.2%}, {ci_high:+.2%}]).",
                f"One-sample t-test vs 0, p={p_value:.4f} (uncorrected - a diagnostic, not itself a promotion "
                "gate; see module docstring)."]

    if abs(mean_diff) < MIN_ECONOMIC_EFFECT:
        return IncrementalValueResult(condition_label, n, mean_diff, p_value, ci_low, ci_high,
                                       "NO_INCREMENTAL_VALUE",
                                       evidence + [f"Incremental effect below the {MIN_ECONOMIC_EFFECT:.1%} "
                                                    "economically-meaningful threshold - the existing model "
                                                    "already captures this; the condition adds nothing beyond it."])

    status = "INCREMENTAL_VALUE_CONFIRMED" if p_value < ALPHA else "NO_INCREMENTAL_VALUE"
    return IncrementalValueResult(condition_label, n, mean_diff, p_value, ci_low, ci_high, status, evidence)
