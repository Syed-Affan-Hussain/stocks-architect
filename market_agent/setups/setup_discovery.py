"""SetupDiscoveryEngine - stage 8's central new capability: searching for
COMPOSITE, multi-concept technical setups directly in the continuous
market-state scan (setups/market_scan.py), rather than testing single
event-conditioned dimensions one at a time (that is what stages 6/7's
learn/hierarchical_research.py already does, and it stays exactly as it
was - this module is a SEPARATE, parallel search over a SEPARATE
population, never a replacement).

DOES NOT COPY INDIVIDUAL TRADERS: every technical dimension searched here
comes from concepts/technical_context.py's OWN computed fields - the same
canonical, fixed vocabulary methodology ingestion (market_agent/methodology/)
independently maps onto. A methodology may have SUGGESTED that "breakout +
volume" is worth looking at, but this engine asks the harder, general
question directly of the data: across the available historical universe,
does breakout-plus-volume (regardless of which, if any, methodology named
it) produce a statistically and economically significant, TEST-surviving
conditional distribution of forward returns? Provenance from methodology
ingestion is not wired into this module at all (a disclosed simplification
for this first version - a future extension could tag a discovered Setup
with which methodology's claim it happens to match, for audit only, never
as evidence, exactly matching the discipline already established for
validated_relationships' concept/methodology_ids_json columns).

HIERARCHICAL, BUDGET-BOUNDED SEARCH - THE SAME PHILOSOPHY AS
learn/hierarchical_research.py, APPLIED TO COMBINATIONS INSTEAD OF SINGLE
DIMENSIONS: Level 1 screens each INDIVIDUAL technical/regime dimension
alone (does it show ANY signal at all, on TRAIN data); only dimensions
that pass Level 1 are even eligible to become a Level 2 CANDIDATE. Level 2
first tests each screened-in dimension's own single most-frequent
non-default value ALONE (a real, promotable single-concept setup in its
own right - not everything interesting requires a composite; see module
docstring above on not copying individual traders), THEN forms
conjunctions of up to `max_dimensions_per_combination` screened-in
dimensions, each again contributing its own single most-frequent
non-default value (a disclosed, outcome-independent choice - the same
"choose by observed frequency, not by which value tests best" discipline
hierarchical_research.py already uses for Level 2/3 setup values), bounded
to `max_combinations_tested` and Holm-corrected together as ONE batch.
This is what keeps "search combinations of concepts" from becoming
unrestricted feature mining: a dimension that shows nothing alone never
gets to combine with anything.

FOUR CHRONOLOGICAL SEGMENTS, NOT threaded through learn/shadow.py's
per-relationship probation window: stage 8's TRAIN/VALIDATE/SHADOW/TEST is
a FIXED, disclosed chronological split of the observation stream itself
(TRAIN_FRACTION/VALIDATE_FRACTION/SHADOW_FRACTION/TEST_FRACTION below),
deliberately named the same way as the rest of this project's evidence
hierarchy. A Setup escalates from TRAIN_SCREENED -> VALIDATED -> SHADOW ->
TEST_VALIDATED only by clearing the SAME significance+economic-effect gate
independently on EACH successive segment's own matching observations, AND
deviating from the baseline in the SAME DIRECTION TRAIN did (see
`_escalates` below) - a segment whose mean is "significantly different
from baseline" in the OPPOSITE direction from TRAIN is a sign flip, not a
genuine re-confirmation, and must not silently escalate just because the
raw t-test cleared ALPHA. Never by re-testing the same data twice, and
never by loosening the gate partway through. Failing any stage is
REJECTED immediately and permanently for that candidate; a later segment
is never even looked at. (Each segment's own raw SetupTestResult is still
reported honestly and unedited either way - only the ESCALATION decision,
not the underlying number, depends on sign-matching TRAIN.)

ONE FIXED BASELINE, COMPUTED ONCE FROM TRAIN, REUSED FOR EVERY SEGMENT'S
TEST: exactly the same discipline as this project's `unconditional_baseline`
(estimated once from burn-in data, never recomputed mid-run) - see
experiment/walkforward.py's `_estimate_baseline`. Comparing VALIDATE/
SHADOW/TEST against a baseline re-estimated from EACH of those segments
would let the comparison point itself drift with whatever's being tested
against it, which is not what "does this setup beat a hard-coded baseline"
is supposed to mean.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

from scipy import stats

from market_agent.concepts.technical_context import TECHNICAL_DEFAULT_VALUES, TECHNICAL_STATE_FIELD_NAMES
from market_agent.learn.hypothesis_testing import ALPHA, MIN_ECONOMIC_EFFECT, MIN_N, _holm_correct
from market_agent.store import db
from market_agent.strategy.decision_process import INVALIDATION_BASELINE_MULTIPLE

# regime is screened/combined exactly like every technical dimension - see module docstring's Level
# 1/2 note. Its own "nothing distinguishing happening" values mirror every other field's convention
# (UNKNOWN/FLAT/NONE-style defaults - see concepts/technical_context.py's TECHNICAL_DEFAULT_VALUES).
REGIME_DEFAULT_VALUES: tuple[str, ...] = ("UNKNOWN", "NORMAL")
SCREENABLE_DIMENSIONS: tuple[str, ...] = ("regime",) + tuple(TECHNICAL_STATE_FIELD_NAMES)

# Fixed, disclosed chronological split of the observation stream - see module docstring. Sums to 1.0.
TRAIN_FRACTION = 0.40
VALIDATE_FRACTION = 0.30
SHADOW_FRACTION = 0.15
TEST_FRACTION = 0.15

SETUP_STATUSES: tuple[str, ...] = ("TRAIN_SCREENED", "VALIDATED", "SHADOW", "TEST_VALIDATED", "REJECTED")


@dataclass(frozen=True)
class SetupSearchBudget:
    """Every field is a hard cap, fixed and recorded BEFORE a discovery
    pass runs - never recalculated or raised after seeing how many
    candidates would otherwise be generated. See module docstring."""
    max_single_dimensions_screened: int
    max_dimensions_per_combination: int
    max_combinations_tested: int
    label: str


DEFAULT_SETUP_SEARCH_BUDGET = SetupSearchBudget(
    max_single_dimensions_screened=len(SCREENABLE_DIMENSIONS),  # 19 - regime + every technical state field
    max_dimensions_per_combination=2,  # conjunctions of exactly 2 screened-in dimensions - bounded, disclosed
    max_combinations_tested=40,
    label="stage8_default_v1",
)


@dataclass
class SetupTestResult:
    segment: str  # "TRAIN" | "VALIDATE" | "SHADOW" | "TEST"
    status: str   # "INSUFFICIENT_N" | "REJECTED_NOT_SIGNIFICANT" | "REJECTED_ECONOMICALLY_TRIVIAL" | "CONFIRMED"
    n: int
    mean_effect: float | None
    baseline: float | None
    p_value: float | None
    p_value_corrected: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"segment": self.segment, "status": self.status, "n": self.n, "mean_effect": self.mean_effect,
                "baseline": self.baseline, "p_value": self.p_value, "p_value_corrected": self.p_value_corrected,
                "ci_low": self.ci_low, "ci_high": self.ci_high, "evidence": self.evidence}


@dataclass
class Setup:
    setup_id: str
    regime: str | None
    technical_conditions: dict[str, str]
    horizon_days: int
    invalidation_pct: float | None
    train_result: SetupTestResult | None = None
    validate_result: SetupTestResult | None = None
    shadow_result: SetupTestResult | None = None
    test_result: SetupTestResult | None = None
    status: str = "TRAIN_SCREENED"

    def to_dict(self) -> dict:
        return {"setup_id": self.setup_id, "regime": self.regime, "technical_conditions": self.technical_conditions,
                "horizon_days": self.horizon_days, "invalidation_pct": self.invalidation_pct,
                "train_result": self.train_result.to_dict() if self.train_result else None,
                "validate_result": self.validate_result.to_dict() if self.validate_result else None,
                "shadow_result": self.shadow_result.to_dict() if self.shadow_result else None,
                "test_result": self.test_result.to_dict() if self.test_result else None, "status": self.status}


@dataclass
class SegmentedObservations:
    train: list[sqlite3.Row]
    validate: list[sqlite3.Row]
    shadow: list[sqlite3.Row]
    test: list[sqlite3.Row]
    train_baseline_mean: float | None


@dataclass
class SetupDiscoveryReport:
    horizon_days: int
    budget: SetupSearchBudget
    n_observations: int
    train_baseline_mean: float | None
    dimensions_screened: int
    dimensions_dropped_by_budget: list[str] = field(default_factory=list)
    level1_results: list[tuple[str, SetupTestResult]] = field(default_factory=list)
    setups: list[Setup] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def segment_observations(rows: list[sqlite3.Row]) -> SegmentedObservations:
    """Chronological TRAIN/VALIDATE/SHADOW/TEST split by `as_of` - see
    module docstring for why the fractions are fixed and why the baseline
    is computed ONLY from TRAIN."""
    ordered = sorted(rows, key=lambda r: r["as_of"])
    n = len(ordered)
    n_train = int(n * TRAIN_FRACTION)
    n_validate = int(n * VALIDATE_FRACTION)
    n_shadow = int(n * SHADOW_FRACTION)
    train = ordered[:n_train]
    validate = ordered[n_train:n_train + n_validate]
    shadow = ordered[n_train + n_validate:n_train + n_validate + n_shadow]
    test = ordered[n_train + n_validate + n_shadow:]
    baseline = (sum(r["realized_abnormal_return"] for r in train) / len(train)) if train else None
    return SegmentedObservations(train, validate, shadow, test, baseline)


def _dimension_value(row: sqlite3.Row, dimension: str) -> str | None:
    if dimension == "regime":
        return row["regime"]
    return json.loads(row["technical_json"]).get(dimension)


def _is_default_value(dimension: str, value) -> bool:
    if dimension == "regime":
        return value in (None,) + REGIME_DEFAULT_VALUES
    return value is None or value in TECHNICAL_DEFAULT_VALUES.get(dimension, ())


def _run_setup_significance_test(result_id: str, matched_rows: list[sqlite3.Row], baseline_mean: float,
                                  segment: str, n_note: str) -> SetupTestResult:
    """Setup-level analogue of learn/hypothesis_testing.py's
    _run_significance_test - NOT reused directly because that function
    hardcodes signing the baseline by `condition["direction"]`, a concept
    that does not exist for a setup (a setup has no event, so no
    positive/negative direction to sign against; its own mean effect can
    come out positive or negative on its own)."""
    n = len(matched_rows)
    if n < MIN_N:
        return SetupTestResult(segment, "INSUFFICIENT_N", n, None, baseline_mean, None,
                                evidence=[f"Only {n} matching {segment} observations {n_note} - below MIN_N={MIN_N}."])

    realized = [r["realized_abnormal_return"] for r in matched_rows]
    diffs = [x - baseline_mean for x in realized]
    t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0)
    mean_effect = sum(realized) / n

    sem = (stats.tstd(realized) / (n ** 0.5)) if n > 1 else 0.0
    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci_low, ci_high = mean_effect - t_crit * sem, mean_effect + t_crit * sem

    evidence = [f"N={n} matching {segment} observations {n_note}.",
                f"{segment} mean realized abnormal return: {mean_effect:+.2%} (95% CI [{ci_low:+.2%}, "
                f"{ci_high:+.2%}]) vs. TRAIN-derived baseline {baseline_mean:+.2%}.",
                f"One-sample t-test of (realized - baseline), p={p_value:.4f} (uncorrected)."]

    if abs(mean_effect - baseline_mean) < MIN_ECONOMIC_EFFECT:
        return SetupTestResult(segment, "REJECTED_ECONOMICALLY_TRIVIAL", n, mean_effect, baseline_mean, p_value,
                                ci_low=ci_low, ci_high=ci_high,
                                evidence=evidence + [f"Effect size below the minimum economically meaningful "
                                                      f"threshold ({MIN_ECONOMIC_EFFECT:.1%})."])
    status = "CONFIRMED" if p_value < ALPHA else "REJECTED_NOT_SIGNIFICANT"
    return SetupTestResult(segment, status, n, mean_effect, baseline_mean, p_value, ci_low=ci_low, ci_high=ci_high,
                            evidence=evidence)


def run_setup_level1_screening(train_rows: list[sqlite3.Row], baseline_mean: float,
                                budget: SetupSearchBudget) -> tuple[list[tuple[str, SetupTestResult]], list[str]]:
    """Every dimension in SCREENABLE_DIMENSIONS up to
    budget.max_single_dimensions_screened is tested ALONE against TRAIN
    (any non-default value pooled together, same "family screening"
    philosophy as learn/hierarchical_research.py's Level 1); the rest are
    recorded as dropped, never silently ignored."""
    dimensions = SCREENABLE_DIMENSIONS[:budget.max_single_dimensions_screened]
    dropped = list(SCREENABLE_DIMENSIONS[budget.max_single_dimensions_screened:])

    raw: list[tuple[str, SetupTestResult]] = []
    for dim in dimensions:
        matched = [r for r in train_rows if not _is_default_value(dim, _dimension_value(r, dim))]
        result = _run_setup_significance_test(f"level1::{dim}", matched, baseline_mean, "TRAIN",
                                               f"(any non-default {dim})")
        raw.append((dim, result))

    testable = [(i, r) for i, (_, r) in enumerate(raw) if r.p_value is not None]
    if testable:
        corrected = _holm_correct([r.p_value for _, r in testable])
        for (i, r), p_adj in zip(testable, corrected):
            r.p_value_corrected = p_adj
            if r.status == "CONFIRMED" and p_adj >= ALPHA:
                r.status = "REJECTED_NOT_SIGNIFICANT"
                r.evidence.append(f"Significant uncorrected (p={r.p_value:.4f}) but NOT after Holm-Bonferroni "
                                   f"correction across {len(testable)} Level-1 dimensions tested together "
                                   f"(corrected p={p_adj:.4f}).")
    return raw, dropped


def _top_value(train_rows: list[sqlite3.Row], dim: str) -> str | None:
    vals = [_dimension_value(r, dim) for r in train_rows if not _is_default_value(dim, _dimension_value(r, dim))]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _build_level2_candidates(train_rows: list[sqlite3.Row], screened_in_dims: list[str],
                              budget: SetupSearchBudget) -> list[dict[str, str]]:
    """Every screened-in dimension's own single MOST-FREQUENT non-default
    value among TRAIN rows is a disclosed, outcome-independent choice (the
    value is picked by which one is common, never by which one tests
    best) - materially more specific than Level 1's "any non-default
    value" pooled test, so worth testing on its own (size 1, a genuine,
    promotable single-concept setup - see module docstring on not
    requiring composites). THEN, size-2..max_dimensions_per_combination
    CONJUNCTIONS among the same screened-in dimensions add the genuinely
    composite candidates the discovery engine exists to explore. Bounded
    to budget.max_combinations_tested throughout, size-1 candidates
    filled first so budget priority favors simpler explanations."""
    candidates: list[dict[str, str]] = []
    for dim in screened_in_dims:
        if len(candidates) >= budget.max_combinations_tested:
            return candidates
        top = _top_value(train_rows, dim)
        if top is not None:
            candidates.append({dim: top})

    for size in range(2, budget.max_dimensions_per_combination + 1):
        for combo in combinations(screened_in_dims, size):
            if len(candidates) >= budget.max_combinations_tested:
                return candidates
            values: dict[str, str] = {}
            ok = True
            for dim in combo:
                top = _top_value(train_rows, dim)
                if top is None:
                    ok = False
                    break
                values[dim] = top
            if ok:
                candidates.append(values)
    return candidates


def _matches(row: sqlite3.Row, conditions: dict[str, str]) -> bool:
    return all(_dimension_value(row, dim) == value for dim, value in conditions.items())


def _escalates(train_result: SetupTestResult, segment_result: SetupTestResult, baseline: float) -> bool:
    """A segment result only counts as RE-confirming what TRAIN found if
    it clears the significance/economic-effect gate AND deviates from the
    baseline in the SAME DIRECTION TRAIN did - "significantly different
    from baseline" in the OPPOSITE direction is a sign flip, not a
    genuine re-confirmation (the same failure mode learn/
    overfitting_diagnostics.py's temporal-stability check exists to catch
    at the event-conditioned level; this is that same discipline applied
    to the TRAIN->VALIDATE->SHADOW->TEST escalation itself)."""
    if segment_result.status != "CONFIRMED" or segment_result.mean_effect is None or train_result.mean_effect is None:
        return False
    return (train_result.mean_effect - baseline) * (segment_result.mean_effect - baseline) > 0


def _split_regime(conditions: dict[str, str]) -> tuple[str | None, dict[str, str]]:
    technical = {k: v for k, v in conditions.items() if k != "regime"}
    return conditions.get("regime"), technical


def run_setup_discovery_pass(conn: sqlite3.Connection, horizon_days: int, created_at: datetime,
                              budget: SetupSearchBudget = DEFAULT_SETUP_SEARCH_BUDGET) -> SetupDiscoveryReport:
    """Top-level orchestrator: TRAIN-only Level 1/2 search, then each
    Level-2-confirmed candidate escalates through VALIDATE -> SHADOW ->
    TEST independently, stopping (permanently REJECTED) the moment it
    fails any one of them. Writes every resulting Setup to
    discovered_setups (db.upsert_discovered_setup) regardless of outcome -
    a REJECTED setup is recorded, never silently dropped."""
    rows = db.query_setup_observations(conn, horizon_days=horizon_days, outcome_known_only=True)
    segmented = segment_observations(rows)
    report = SetupDiscoveryReport(horizon_days=horizon_days, budget=budget, n_observations=len(rows),
                                   train_baseline_mean=segmented.train_baseline_mean, dimensions_screened=0,
                                   evidence=[f"SetupSearchBudget recorded before execution: {budget}"])

    if segmented.train_baseline_mean is None or len(segmented.train) < MIN_N:
        report.evidence.append(f"Only {len(segmented.train)} TRAIN observations - below MIN_N={MIN_N}. "
                                "No discovery attempted.")
        return report

    level1_results, dropped = run_setup_level1_screening(segmented.train, segmented.train_baseline_mean, budget)
    report.level1_results = level1_results
    report.dimensions_screened = len(level1_results)
    report.dimensions_dropped_by_budget = dropped
    if dropped:
        report.evidence.append(f"{len(dropped)} dimensions dropped by SetupSearchBudget before Level 1 even ran: "
                                f"{dropped}.")

    screened_in = [dim for dim, r in level1_results if r.status == "CONFIRMED"]
    report.evidence.append(f"Level 1: {len(screened_in)}/{len(level1_results)} dimensions screened in "
                            f"(TRAIN, vs. baseline {segmented.train_baseline_mean:+.2%}): {screened_in}")

    candidates = _build_level2_candidates(segmented.train, screened_in, budget)
    level2_results = [(cond, _run_setup_significance_test(f"level2::{cond}",
                                                            [r for r in segmented.train if _matches(r, cond)],
                                                            segmented.train_baseline_mean, "TRAIN", f"(combo {cond})"))
                       for cond in candidates]
    testable = [(i, r) for i, (_, r) in enumerate(level2_results) if r.p_value is not None]
    if testable:
        corrected = _holm_correct([r.p_value for _, r in testable])
        for (i, r), p_adj in zip(testable, corrected):
            r.p_value_corrected = p_adj
            if r.status == "CONFIRMED" and p_adj >= ALPHA:
                r.status = "REJECTED_NOT_SIGNIFICANT"
                r.evidence.append(f"Significant uncorrected but NOT after Holm-Bonferroni correction across "
                                   f"{len(testable)} Level-2 combinations tested together (corrected p={p_adj:.4f}).")
    report.evidence.append(f"Level 2: {len(candidates)} combination(s) tested, "
                            f"{sum(1 for _, r in level2_results if r.status == 'CONFIRMED')} confirmed on TRAIN.")

    for conditions, train_result in level2_results:
        regime, technical = _split_regime(conditions)
        setup_id = f"S_{uuid.uuid4().hex[:10]}"
        baseline_mag = segmented.train_baseline_mean
        invalidation_pct = INVALIDATION_BASELINE_MULTIPLE * abs(baseline_mag) if baseline_mag is not None else None
        setup = Setup(setup_id=setup_id, regime=regime, technical_conditions=technical, horizon_days=horizon_days,
                      invalidation_pct=invalidation_pct, train_result=train_result, status="REJECTED")

        if train_result.status != "CONFIRMED":
            report.setups.append(setup)
            _persist_setup(conn, setup, created_at)
            continue
        setup.status = "TRAIN_SCREENED"

        validate_matched = [r for r in segmented.validate if _matches(r, conditions)]
        setup.validate_result = _run_setup_significance_test(f"validate::{conditions}", validate_matched,
                                                               segmented.train_baseline_mean, "VALIDATE",
                                                               f"(combo {conditions})")
        if not _escalates(train_result, setup.validate_result, segmented.train_baseline_mean):
            setup.status = "REJECTED"
            report.setups.append(setup)
            _persist_setup(conn, setup, created_at)
            continue
        setup.status = "VALIDATED"

        shadow_matched = [r for r in segmented.shadow if _matches(r, conditions)]
        setup.shadow_result = _run_setup_significance_test(f"shadow::{conditions}", shadow_matched,
                                                             segmented.train_baseline_mean, "SHADOW",
                                                             f"(combo {conditions})")
        if not _escalates(train_result, setup.shadow_result, segmented.train_baseline_mean):
            setup.status = "REJECTED"
            report.setups.append(setup)
            _persist_setup(conn, setup, created_at)
            continue
        setup.status = "SHADOW"

        test_matched = [r for r in segmented.test if _matches(r, conditions)]
        setup.test_result = _run_setup_significance_test(f"test::{conditions}", test_matched,
                                                           segmented.train_baseline_mean, "TEST",
                                                           f"(combo {conditions})")
        setup.status = ("TEST_VALIDATED" if _escalates(train_result, setup.test_result, segmented.train_baseline_mean)
                         else "REJECTED")
        report.setups.append(setup)
        _persist_setup(conn, setup, created_at)

    n_test_validated = sum(1 for s in report.setups if s.status == "TEST_VALIDATED")
    report.evidence.append(f"{n_test_validated}/{len(report.setups)} Level-2 candidate(s) reached TEST_VALIDATED.")
    return report


def _persist_setup(conn: sqlite3.Connection, setup: Setup, created_at: datetime) -> None:
    db.upsert_discovered_setup(
        conn, setup.setup_id, setup.regime, setup.technical_conditions, setup.horizon_days, setup.invalidation_pct,
        train_result=setup.train_result.to_dict() if setup.train_result else None,
        validate_result=setup.validate_result.to_dict() if setup.validate_result else None,
        shadow_result=setup.shadow_result.to_dict() if setup.shadow_result else None,
        test_result=setup.test_result.to_dict() if setup.test_result else None,
        status=setup.status, created_at=created_at,
    )
