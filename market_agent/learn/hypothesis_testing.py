"""Testing a candidate hypothesis against prior, held-out history -
Blueprint sections G and Q. This module has no LLM involvement at all and
is the actual gate between "someone proposed an idea" and "the system's
live predictions changed."

HINDSIGHT-BIAS DEFENSE: a hypothesis is tested only against episodic_events
rows published STRICTLY BEFORE the event that spawned it (`source_event`'s
own published_at is excluded from the matched sample, and nothing
published after it is visible either via the `published_before` gate).
Testing a hypothesis against the very case that inspired it - or against
data from after that case - is exactly the hindsight-fitting the blueprint
warns about; excluding both is what makes "supported by prior evidence"
a meaningful phrase here rather than a tautology.

MULTIPLE-TESTING CORRECTION: Holm-Bonferroni across whichever hypotheses
are tested together in one batch (test_hypotheses_batch). This is applied
at the hypothesis level; Blueprint section Q's SYSTEM-level correction
(across different learning-mechanism variants) is a separate, higher-level
concern for the experiment harness, not this module.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from scipy import stats

from market_agent.store.db import deduplicate_by_real_event

MIN_N = 15                    # minimum supporting prior observations before a test is even attempted
MIN_ECONOMIC_EFFECT = 0.003   # 0.3% - a statistically significant but economically trivial effect is not promoted
ALPHA = 0.05


@dataclass
class HypothesisTestResult:
    hypothesis_id: str
    status: str  # "REJECTED_INSUFFICIENT_N" | "REJECTED_NOT_SIGNIFICANT" | "REJECTED_ECONOMICALLY_TRIVIAL" | "CONFIRMED"
    n: int
    mean_effect: float | None
    baseline_effect: float | None
    p_value: float | None
    p_value_corrected: float | None
    ci_low: float | None = None    # 95% CI on mean_effect (t-distribution) - stage 4 addition, defaulted
    ci_high: float | None = None   # so existing positional/keyword construction elsewhere keeps working
    evidence: list[str] = field(default_factory=list)


def _matching_prior_rows(conn: sqlite3.Connection, condition: dict, horizon_days: int,
                          published_before: str, exclude_event_id: str | None = None) -> list[sqlite3.Row]:
    """`exclude_event_id=None` (the revalidation case, learn/revalidation.py) means
    'no single triggering event to exclude' - every matching, already-resolved case
    up to `published_before` counts. A brand-new hypothesis (learn/hypothesis_testing.
    test_hypothesis) always passes its own source event's id here instead.

    `outcome_locked = 1` alone is NOT enough to mean 'has a usable numeric outcome' -
    a row is locked (and permanently so - see store/db.py's AppendOnlyViolation) the
    moment ANY outcome is recorded, including a DATA_ERROR outcome (e.g. a delisted
    ticker with no price history at the horizon date), whose realized_abnormal_return
    is NULL by design (outcomes/observe.py never fabricates a return from missing
    price data). Found by running against real Yahoo Finance data, where a real
    fraction of tickers are genuinely delisted/renamed within the experiment window -
    `realized_abnormal_return IS NOT NULL` is required explicitly rather than assumed
    from the lock flag."""
    if exclude_event_id is not None:
        rows = conn.execute(
            """SELECT * FROM episodic_events
               WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL AND horizon_days = ?
                 AND event_type = ? AND direction = ?
                 AND published_at < ? AND event_id != ?""",
            (horizon_days, condition["event_type"], condition["direction"], published_before, exclude_event_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM episodic_events
               WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL AND horizon_days = ?
                 AND event_type = ? AND direction = ? AND published_at < ?""",
            (horizon_days, condition["event_type"], condition["direction"], published_before),
        ).fetchall()
    extra_keys = {k: v for k, v in condition.items() if k not in ("event_type", "direction")}
    if extra_keys:
        rows = [r for r in rows if all(json.loads(r["context_json"]).get(k) == v for k, v in extra_keys.items())]
    # CRITICAL: dedupe by real underlying event BEFORE returning - see
    # store/db.py::deduplicate_by_real_event's docstring. Without this, N (and the t-test's standard
    # error) is inflated by however many agents logged a prediction for the same real event.
    return deduplicate_by_real_event(rows)


def test_hypothesis(conn: sqlite3.Connection, hypothesis_row: sqlite3.Row,
                     unconditional_baseline: dict[int, float]) -> HypothesisTestResult:
    """Single-hypothesis test, uncorrected p-value only - use
    test_hypotheses_batch for the corrected version actually used to gate
    promotion. Exposed separately because the batch function needs each
    individual p-value before it can apply Holm correction across them."""
    hid = hypothesis_row["hypothesis_id"]
    condition = json.loads(hypothesis_row["condition_json"])
    horizon_days = hypothesis_row["horizon_days"]
    source_event = conn.execute("SELECT * FROM episodic_events WHERE event_id = ?",
                                 (hypothesis_row["source_event_id"],)).fetchone()

    prior_rows = _matching_prior_rows(conn, condition, horizon_days, source_event["published_at"],
                                       exclude_event_id=hypothesis_row["source_event_id"])
    n = len(prior_rows)
    if n < MIN_N:
        return HypothesisTestResult(hid, "REJECTED_INSUFFICIENT_N", n, None, None, None, None,
                                     [f"Only {n} prior matching observations found (before "
                                      f"{source_event['published_at']}, excluding the triggering event) - "
                                      f"below the minimum of {MIN_N}. Not enough evidence either way."])

    return _run_significance_test(hid, prior_rows, condition, unconditional_baseline.get(horizon_days, 0.0),
                                   f"(published before {source_event['published_at']}, excluding the "
                                   "triggering event)")


def _run_significance_test(result_id: str, prior_rows: list[sqlite3.Row], condition: dict,
                            baseline: float, n_note: str) -> HypothesisTestResult:
    """Shared statistical core for both a brand-new hypothesis test
    (test_hypothesis, above) and a periodic revalidation test
    (test_relationship, below) - same significance test, same economic-
    effect gate, same evidence format, whichever direction the caller is
    coming from."""
    n = len(prior_rows)
    signed_baseline = baseline if condition["direction"] == "positive" else -baseline
    realized = [r["realized_abnormal_return"] for r in prior_rows]
    diffs = [x - signed_baseline for x in realized]

    t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0)
    mean_effect = sum(realized) / n

    # 95% CI on mean_effect itself (not on the diff-from-baseline) - t-distribution, n-1 df. Reported
    # alongside the point estimate everywhere this estimate is surfaced (knowledge-state report,
    # prediction ledger's `uncertainty` field) rather than only ever showing a bare number.
    sem = (stats.tstd(realized) / (n ** 0.5)) if n > 1 else 0.0
    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci_low, ci_high = mean_effect - t_crit * sem, mean_effect + t_crit * sem

    evidence = [f"N={n} prior matching cases {n_note}.",
                f"Segment mean realized abnormal return: {mean_effect:+.2%} (95% CI [{ci_low:+.2%}, "
                f"{ci_high:+.2%}]) vs. unconditional baseline {signed_baseline:+.2%}.",
                f"One-sample t-test of (realized - baseline), p={p_value:.4f} (uncorrected)."]

    if abs(mean_effect - signed_baseline) < MIN_ECONOMIC_EFFECT:
        return HypothesisTestResult(result_id, "REJECTED_ECONOMICALLY_TRIVIAL", n, mean_effect, signed_baseline,
                                     p_value, None, ci_low=ci_low, ci_high=ci_high,
                                     evidence=evidence + ["Effect size below the minimum economically "
                                     f"meaningful threshold ({MIN_ECONOMIC_EFFECT:.1%})."])

    return HypothesisTestResult(result_id, "CONFIRMED" if p_value < ALPHA else "REJECTED_NOT_SIGNIFICANT",
                                 n, mean_effect, signed_baseline, p_value, None, ci_low=ci_low, ci_high=ci_high,
                                 evidence=evidence)


def test_relationship(conn: sqlite3.Connection, relationship_row: sqlite3.Row,
                       unconditional_baseline: dict[int, float], as_of) -> HypothesisTestResult:
    """Periodic revalidation (Blueprint section M / learn/revalidation.py):
    re-tests an already-ACTIVE validated_relationship against ALL matching
    resolved history up to `as_of` - no single triggering event to
    exclude here, unlike a brand-new hypothesis. Same significance test
    and economic-effect gate as a first-time test; no leniency for
    something already promoted."""
    condition = json.loads(relationship_row["condition_json"])
    horizon_days = relationship_row["horizon_days"]
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else as_of
    prior_rows = _matching_prior_rows(conn, condition, horizon_days, as_of_iso)
    n = len(prior_rows)
    if n < MIN_N:
        return HypothesisTestResult(relationship_row["relationship_id"], "REJECTED_INSUFFICIENT_N", n,
                                     None, None, None, None,
                                     [f"Only {n} matching observations found as of {as_of_iso} - below the "
                                      f"minimum of {MIN_N}. Cannot confirm the relationship still holds, but "
                                      "also cannot conclude it doesn't - insufficient evidence either way."])
    return _run_significance_test(relationship_row["relationship_id"], prior_rows, condition,
                                   unconditional_baseline.get(horizon_days, 0.0), f"as of {as_of_iso}")


def _holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the ORIGINAL input order. Standard, conservative, and - unlike a
    single Bonferroni divide - doesn't needlessly over-penalize when only
    one or two of many tested hypotheses are actually significant."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted


def test_hypotheses_batch(conn: sqlite3.Connection, hypothesis_rows: list[sqlite3.Row],
                           unconditional_baseline: dict[int, float]) -> list[HypothesisTestResult]:
    """The actual entry point used to gate promotion. Every hypothesis
    tested in the same batch counts as one trial in the SAME
    Holm-Bonferroni family - testing 20 hypotheses and reporting the one
    that happened to clear an UNCORRECTED p<0.05 is precisely the
    selection bias Blueprint section Q exists to prevent."""
    results = [test_hypothesis(conn, row, unconditional_baseline) for row in hypothesis_rows]

    testable = [(i, r) for i, r in enumerate(results) if r.p_value is not None]
    if testable:
        corrected = _holm_correct([r.p_value for _, r in testable])
        for (i, r), p_adj in zip(testable, corrected):
            r.p_value_corrected = p_adj
            if r.status == "CONFIRMED" and p_adj >= ALPHA:
                r.status = "REJECTED_NOT_SIGNIFICANT"
                r.evidence.append(f"Significant uncorrected (p={r.p_value:.4f}) but NOT after Holm-Bonferroni "
                                   f"correction across {len(testable)} hypotheses tested in this batch "
                                   f"(corrected p={p_adj:.4f}).")
    return results
