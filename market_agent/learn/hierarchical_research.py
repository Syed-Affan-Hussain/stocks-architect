"""Hierarchical, budget-bounded research procedure - stage 7's replacement
for technical/methodology concept DISCOVERY. Stage 6's real run showed the
flat, reactive, bounded-powerset approach (learn/hypothesis.py's
RuleBasedHypothesisGenerator with include_technical_dimensions=True)
generates so many simultaneously-tested candidates (77,190 hypotheses in
one run) that Holm-Bonferroni correction becomes severe enough that ZERO
technical-concept hypotheses were ever confirmed, regardless of whether
any of them contained real signal. This module does not loosen that
correction - it changes the SEARCH STRATEGY so far fewer candidates ever
reach a shared correction batch in the first place, using a real,
standard statistical design (fixed-sequence / gatekeeping hierarchical
testing - see e.g. clinical-trials multiple-endpoint literature), not an
ad hoc shrinking of batch sizes for convenience.

FOUR SEQUENTIAL, GATED LEVELS:

  Level 1 - FAMILY (methodology) screening: does this canonical trading
    concept (concepts/ontology.py) show ANY signal at all, pooling every
    one of its "interesting" (non-default) technical states together into
    ONE test? At most one test per computable concept (17 total, capped by
    ResearchBudget.max_level1_families), corrected together as ONE small
    batch. A family that fails here NEVER spawns a single Level 2
    candidate - this is what keeps the total candidate count bounded, not
    an arity cap alone. Level 1 produces no promotable relationship of its
    own (see run_level1_family_screening) - it is a pure screening gate.

  Level 2 - SETUP validation: for each family that passed Level 1, test
    its SPECIFIC observed values (e.g. BREAKOUT_UP vs BREAKOUT_DOWN) as
    individual setups - up to ResearchBudget.max_level2_setups_per_family,
    chosen by observed frequency (a disclosed, non-cherry-picking
    selection decided from the DATA'S OWN distribution before testing, not
    from which setup looks most promising after testing). Corrected
    together as one batch PER FAMILY, never globally across all families -
    a real, standard hierarchical/gatekeeping design.

  Level 3 - CONTEXT conditioning: for each setup CONFIRMED at Level 2,
    test whether ONE existing event-context dimension at a time (regime,
    prior_return_bucket, vol_bucket - the SAME three dimensions stage 1-5
    already use) materially changes its expectancy - again a small,
    separate batch per setup, up to
    ResearchBudget.max_level3_context_dims_per_setup, values again chosen
    by observed frequency.

  Level 4 - PARAMETER refinement (bounded grid search on an already-
    Level-1-3-confirmed condition, using TRAIN data only) - NOT YET BUILT.
    Disclosed gap, not a silent omission: stage 7 prioritized the
    TRAIN/VALIDATE/SHADOW/TEST discipline (experiment/four_way_walkforward.py's
    freeze_governance_during_test) and the anti-overfitting diagnostics
    below over this level.

EVERY LEVEL 2/3 RESULT ALSO CARRIES TWO ADDITIONAL, REPORT-ONLY
DIAGNOSTICS (LevelTestResult, below): learn/incremental_value.py (does
this beat CURRENT_ADAPTIVE's own prediction, not just a fixed baseline
scalar) and learn/overfitting_diagnostics.py (a permutation test against
random same-size subsets of the same pool, and a temporal-stability check
across the condition's own history). NONE of the three are wired into
apply_test_results - SHADOW entry is decided entirely by the existing,
unchanged significance test against unconditional_baseline.

NOTHING ABOUT THE DOWNSTREAM GOVERNANCE MACHINERY CHANGES: every Level 2/3
candidate is still formalized as an ordinary candidate_hypotheses row
(written before testing), tested via learn/hypothesis_testing.py's
UNCHANGED significance test (same MIN_N, same ALPHA, same economic-effect
gate), and CONFIRMED ones still enter SHADOW (never ACTIVE directly) via
the UNCHANGED learn/governance.py/learn/shadow.py path. Level 1 is the one
genuinely new mechanism, and it never writes to validated_relationships.

POINT-IN-TIME: a Level 2/3 test's "source event" is the LATEST matching
row as of `published_before` - the same test_hypothesis()/
_matching_prior_rows() machinery this project already uses then naturally
tests everything published strictly before that latest occurrence,
preserving the hindsight-bias defense (learn/hypothesis_testing.py's own
module docstring) without any new leakage-defense code.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from market_agent.concepts.technical_context import DIMENSION_TO_CONCEPT, TECHNICAL_DEFAULT_VALUES
from market_agent.learn.governance import apply_test_results
from market_agent.learn.hypothesis import TECHNICAL_DIMENSION_PRIORITY
from market_agent.learn.hypothesis_testing import (
    ALPHA, MIN_N, HypothesisTestResult, _holm_correct, _run_significance_test, test_hypotheses_batch,
)
from market_agent.learn.incremental_value import IncrementalValueResult, test_incremental_value
from market_agent.learn.overfitting_diagnostics import (
    PermutationTestResult, TemporalStabilityResult, _full_pool_rows, run_permutation_test,
    run_temporal_stability_check,
)
from market_agent.store import db

CONTEXT_DIMENSIONS_FOR_LEVEL3: tuple[str, ...] = ("regime", "prior_return_bucket", "vol_bucket")


@dataclass(frozen=True)
class ResearchBudget:
    """Every field here is a hard cap, fixed and recorded BEFORE a
    research pass runs (the caller logs it to the run's evidence text
    before calling run_hierarchical_research_pass) - never recalculated or
    raised after seeing how many candidates would otherwise be generated.
    A family/setup/context list exceeding its cap is truncated by a FIXED
    priority (TECHNICAL_DIMENSION_PRIORITY for families, observed
    frequency for setups/context values - both decided independently of
    any test OUTCOME), and the truncation count is recorded, never
    silently dropped."""
    max_level1_families: int
    max_level2_setups_per_family: int
    max_level3_context_dims_per_setup: int
    label: str


DEFAULT_RESEARCH_BUDGET = ResearchBudget(
    max_level1_families=18,             # every technical dimension with a real state field - see
    #                                      concepts/technical_context.py's TECHNICAL_STATE_FIELD_NAMES
    #                                      (VOLUME and CATALYST_EVENT_REACTION are computable concepts
    #                                      but have no dedicated state field to screen - see that
    #                                      module's docstring; stage 7 item 7 added 3 more fields,
    #                                      15 -> 18) - none dropped a priori
    max_level2_setups_per_family=4,     # at most 4 specific setups per family
    max_level3_context_dims_per_setup=3,  # at most all 3 existing event-context dimensions
    label="stage7_default_v1",
)


@dataclass
class FamilyScreeningResult:
    concept: str
    dimension: str
    test_result: HypothesisTestResult


@dataclass
class LevelTestResult:
    """Bundles the EXISTING, unchanged significance test (vs. the fixed
    unconditional_baseline scalar - this is what actually gates SHADOW
    entry via apply_test_results, unchanged) with the ADDITIONAL,
    separately-reported incremental-value diagnostic (vs. CURRENT_ADAPTIVE's
    own prediction for the same cases - learn/incremental_value.py, never
    itself a promotion gate). A setup/context result can be CONFIRMED on
    the first and still show NO_INCREMENTAL_VALUE on the second - that
    combination is the honest signal that the existing model already
    captured what this condition offers, even though it differs from a
    crude population-average scalar.

    `condition` (stage 7 item 9) is the FULL condition dict this result was
    tested against (event_type/direction plus whichever setup/regime keys
    apply) - carried through so a downstream consumer (reporting/
    stage7_final_report.py) can look up the resulting validated_relationships
    row directly, without re-deriving the condition from a formatted string
    key."""
    test_result: HypothesisTestResult
    incremental_value: IncrementalValueResult | None  # None only if setup_condition/value had zero matching rows
    permutation_test: PermutationTestResult | None
    temporal_stability: TemporalStabilityResult | None
    condition: dict = field(default_factory=dict)


@dataclass
class HierarchicalResearchReport:
    event_type: str
    direction: str
    horizon_days: int
    budget: ResearchBudget
    families_screened: int
    families_dropped_by_budget: list[str] = field(default_factory=list)
    level1_results: list[FamilyScreeningResult] = field(default_factory=list)
    level2_results: dict[str, list[LevelTestResult]] = field(default_factory=dict)   # dimension -> results
    level3_results: dict[str, list[LevelTestResult]] = field(default_factory=dict)   # setup condition key -> results
    evidence: list[str] = field(default_factory=list)


def _matching_rows_any_interesting_value(conn: sqlite3.Connection, dimension_name: str, event_type: str,
                                          direction: str, horizon_days: int,
                                          published_before: str) -> list[sqlite3.Row]:
    default_values = TECHNICAL_DEFAULT_VALUES[dimension_name]
    rows = conn.execute(
        """SELECT * FROM episodic_events
           WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL AND horizon_days = ?
             AND event_type = ? AND direction = ? AND published_at < ?""",
        (horizon_days, event_type, direction, published_before)).fetchall()
    matched = []
    for r in rows:
        value = json.loads(r["context_json"]).get(dimension_name)
        if value is not None and value not in default_values:
            matched.append(r)
    # CRITICAL: same fix as learn/hypothesis_testing.py::_matching_prior_rows - without this, N is
    # inflated by however many agents logged a prediction for the same real event.
    return db.deduplicate_by_real_event(matched)


def run_level1_family_screening(conn: sqlite3.Connection, event_type: str, direction: str, horizon_days: int,
                                 published_before: str, unconditional_baseline: dict[int, float],
                                 budget: ResearchBudget) -> tuple[list[FamilyScreeningResult], list[str]]:
    """Returns (results, families_dropped_by_budget). Every family in
    TECHNICAL_DIMENSION_PRIORITY up to budget.max_level1_families is
    tested; the rest are recorded as dropped, never silently ignored."""
    dimensions = TECHNICAL_DIMENSION_PRIORITY[:budget.max_level1_families]
    dropped = TECHNICAL_DIMENSION_PRIORITY[budget.max_level1_families:]

    raw: list[FamilyScreeningResult] = []
    for dimension in dimensions:
        rows = _matching_rows_any_interesting_value(conn, dimension, event_type, direction, horizon_days,
                                                      published_before)
        n = len(rows)
        concept = DIMENSION_TO_CONCEPT[dimension].value
        if n < MIN_N:
            result = HypothesisTestResult(
                f"level1::{dimension}", "REJECTED_INSUFFICIENT_N", n, None, None, None, None,
                evidence=[f"Only {n} prior cases with any non-default {dimension} value (published before "
                          f"{published_before}) - below MIN_N={MIN_N}."])
        else:
            condition = {"event_type": event_type, "direction": direction}
            result = _run_significance_test(f"level1::{dimension}", rows, condition,
                                             unconditional_baseline.get(horizon_days, 0.0),
                                             f"(any non-default {dimension}, published before {published_before})")
        raw.append(FamilyScreeningResult(concept=concept, dimension=dimension, test_result=result))

    testable = [(i, fr.test_result) for i, fr in enumerate(raw) if fr.test_result.p_value is not None]
    if testable:
        corrected = _holm_correct([r.p_value for _, r in testable])
        for (i, r), p_adj in zip(testable, corrected):
            r.p_value_corrected = p_adj
            if r.status == "CONFIRMED" and p_adj >= ALPHA:
                r.status = "REJECTED_NOT_SIGNIFICANT"
                r.evidence.append(f"Significant uncorrected (p={r.p_value:.4f}) but NOT after Holm-Bonferroni "
                                   f"correction across {len(testable)} Level-1 FAMILIES tested together "
                                   f"(corrected p={p_adj:.4f}) - a small batch by design, not the thousands of "
                                   "combinations stage 6's flat approach corrected across.")
    return raw, dropped


def _top_observed_values(rows: list[sqlite3.Row], dimension_name: str, limit: int) -> list[str]:
    counts = Counter(json.loads(r["context_json"]).get(dimension_name) for r in rows)
    counts.pop(None, None)
    return [value for value, _ in counts.most_common(limit)]


def _formalize_and_test(conn: sqlite3.Connection, condition: dict, horizon_days: int, matching_rows: list[sqlite3.Row],
                         concept: str, explanation: str, proposed_at: datetime, unconditional_baseline: dict[int, float],
                         promoted_by: str, published_before: str) -> LevelTestResult | None:
    """Shared plumbing for Level 2/3: picks the LATEST matching row as the
    nominal source event (see module docstring for why this preserves
    point-in-time correctness with zero new leakage-defense code), writes
    ONE candidate_hypotheses row, tests it via the UNCHANGED
    test_hypotheses_batch/apply_test_results pipeline - this ALONE still
    decides SHADOW entry, exactly as before. ADDITIONALLY runs the
    incremental-value, permutation, and temporal-stability diagnostics
    against the SAME matching_rows and bundles all four into a
    LevelTestResult - none of the three extra tests ever influence
    apply_test_results. Returns None if there's no matching row at all
    (nothing to test)."""
    if not matching_rows:
        return None
    latest = max(matching_rows, key=lambda r: r["published_at"])
    hid = db.add_hypothesis(conn, source_event_id=latest["event_id"], condition=condition, horizon_days=horizon_days,
                             explanation_text=explanation, proposed_at=proposed_at, concept=concept)
    hyp_row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()
    results = test_hypotheses_batch(conn, [hyp_row], unconditional_baseline)
    apply_test_results(conn, results, promoted_by=promoted_by, clock_now=proposed_at)

    incremental = test_incremental_value(conn, condition, horizon_days, matching_rows)
    pool_rows = _full_pool_rows(conn, condition["event_type"], condition["direction"], horizon_days, published_before)
    permutation = run_permutation_test(matching_rows, pool_rows)
    stability = run_temporal_stability_check(matching_rows)
    return LevelTestResult(test_result=results[0], incremental_value=incremental,
                            permutation_test=permutation, temporal_stability=stability, condition=condition)


def run_hierarchical_research_pass(conn: sqlite3.Connection, event_type: str, direction: str, horizon_days: int,
                                    published_before: str, unconditional_baseline: dict[int, float],
                                    proposed_at: datetime, promoted_by: str,
                                    budget: ResearchBudget = DEFAULT_RESEARCH_BUDGET) -> HierarchicalResearchReport:
    report = HierarchicalResearchReport(event_type=event_type, direction=direction, horizon_days=horizon_days,
                                         budget=budget, families_screened=0,
                                         evidence=[f"ResearchBudget recorded before execution: {budget}"])

    level1_results, dropped = run_level1_family_screening(conn, event_type, direction, horizon_days,
                                                            published_before, unconditional_baseline, budget)
    report.level1_results = level1_results
    report.families_dropped_by_budget = dropped
    report.families_screened = len(level1_results)
    if dropped:
        report.evidence.append(f"{len(dropped)} families dropped by ResearchBudget before Level 1 even ran: "
                                f"{dropped} (fixed priority order, not by expected promise).")

    confirmed_families = [fr for fr in level1_results if fr.test_result.status == "CONFIRMED"]
    report.evidence.append(f"Level 1: {len(confirmed_families)}/{len(level1_results)} families screened in.")

    for family_result in confirmed_families:
        dimension = family_result.dimension
        concept = family_result.concept
        rows = _matching_rows_any_interesting_value(conn, dimension, event_type, direction, horizon_days,
                                                      published_before)
        setups = _top_observed_values(rows, dimension, budget.max_level2_setups_per_family)

        level2_results = []
        for value in setups:
            setup_rows = [r for r in rows if json.loads(r["context_json"]).get(dimension) == value]
            condition = {"event_type": event_type, "direction": direction, dimension: value}
            explanation = (f"Level 2 setup validation for family {concept} ({dimension}={value!r}) - "
                            f"tested after {concept} passed Level 1 family screening. Audit prose only.")
            result = _formalize_and_test(conn, condition, horizon_days, setup_rows, concept, explanation,
                                          proposed_at, unconditional_baseline, promoted_by, published_before)
            if result is not None:
                level2_results.append((value, result))
        report.level2_results[dimension] = [r for _, r in level2_results]

        confirmed_setups = [(value, r) for value, r in level2_results if r.test_result.status == "CONFIRMED"]
        n_also_incremental = sum(1 for _, r in confirmed_setups
                                  if r.incremental_value is not None
                                  and r.incremental_value.status == "INCREMENTAL_VALUE_CONFIRMED")
        report.evidence.append(f"  Level 2 ({concept}/{dimension}): {len(confirmed_setups)}/{len(level2_results)} "
                                f"setups confirmed vs. baseline scalar, {n_also_incremental} of those ALSO show "
                                "incremental value vs. CURRENT_ADAPTIVE's own prediction.")

        for setup_value, setup_result in confirmed_setups:
            setup_condition_key = f"{dimension}={setup_value}"
            setup_rows = [r for r in rows if json.loads(r["context_json"]).get(dimension) == setup_value]
            level3_results = []
            for context_dim in CONTEXT_DIMENSIONS_FOR_LEVEL3[:budget.max_level3_context_dims_per_setup]:
                context_values = _top_observed_values(setup_rows, context_dim, 1)  # single most-common value
                if not context_values:
                    continue
                context_value = context_values[0]
                context_rows = [r for r in setup_rows
                                 if json.loads(r["context_json"]).get(context_dim) == context_value]
                condition = {"event_type": event_type, "direction": direction, dimension: setup_value,
                             context_dim: context_value}
                explanation = (f"Level 3 context conditioning for setup {setup_condition_key} under "
                                f"{context_dim}={context_value!r} - tested after the setup passed Level 2. "
                                "Audit prose only.")
                result = _formalize_and_test(conn, condition, horizon_days, context_rows, concept, explanation,
                                              proposed_at, unconditional_baseline, promoted_by, published_before)
                if result is not None:
                    level3_results.append(result)
            report.level3_results[setup_condition_key] = level3_results
            confirmed_context = sum(1 for r in level3_results if r.test_result.status == "CONFIRMED")
            n_also_incremental = sum(1 for r in level3_results
                                      if r.test_result.status == "CONFIRMED" and r.incremental_value is not None
                                      and r.incremental_value.status == "INCREMENTAL_VALUE_CONFIRMED")
            report.evidence.append(f"    Level 3 ({setup_condition_key}): {confirmed_context}/{len(level3_results)} "
                                    f"context-conditioned variants confirmed vs. baseline scalar, "
                                    f"{n_also_incremental} of those ALSO show incremental value.")

    return report
