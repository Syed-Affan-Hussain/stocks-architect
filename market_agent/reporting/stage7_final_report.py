"""Stage 7 items 8+9: the final knowledge-state report over everything
items 1-7 built.

ITEM 8 - THE EVIDENCE HIERARCHY IS ENFORCED IN CODE, NOT JUST DESCRIBED:
candidate -> statistical evidence -> incremental value -> temporal
stability -> economic value -> strategy robustness -> final TEST. Each
stage below only RUNS if the previous one was cleared - a relationship
that fails Level 2/3's baseline-scalar significance test never gets an
incremental-value check attempted; one that fails incremental value never
gets a StrategyAgent built for it; one whose VALIDATE-segment economics
don't clear the bar never has its TEST-segment performance looked at at
all. This short-circuiting IS the enforcement: there is no code path by
which an attractive later-stage number can retroactively promote a
candidate that already failed an earlier gate, because the later-stage
code for that candidate never executes.

ITEM 9 - SIX-STATE TAXONOMY (never collapsed into one "successful" label):

  DISCOVERED            - proposed as a Level 2/3 candidate and formally
                           tested; nothing more is claimed.
  STATISTICALLY_SUPPORTED - cleared learn/hypothesis_testing.py's existing,
                           unchanged significance test (vs. the fixed
                           unconditional-baseline scalar).
  INCREMENTAL           - ALSO beats CURRENT_ADAPTIVE's own logged
                           prediction for the same real events
                           (learn/incremental_value.py).
  SHADOW                 - ALSO survives the permutation test and the
                           temporal-stability check (learn/
                           overfitting_diagnostics.py) - this is exactly
                           the real validated_relationships status
                           apply_test_results already assigned it, not a
                           new database state.
  ECONOMICALLY_SUPPORTED - a StrategyAgent built ONLY from this
                           relationship (strategy/decision_process.py,
                           strategy/strategy_agent.py) produces real
                           VALIDATE-segment trades (strategy/outcome_engine.py)
                           with positive, transaction-cost-adjusted,
                           bootstrap-confirmed expectancy.
  TEST_VALIDATED          - the SAME frozen decision process/StrategyAgent,
                           never reconstructed or re-tuned, retains
                           positive, bootstrap-confirmed economics on the
                           untouched TEST segment.

A candidate's `reached_state` is the LAST stage it cleared; a
`rejection_reason` records exactly why it stopped there. Nothing here
writes to validated_relationships or influences governance - this module
is read-only over what items 1-7 already computed and decided.

TWO-PHASE TEST-ISOLATION DISCIPLINE, NOT PER-RELATIONSHIP: a single run
evaluates MANY candidates. Calling TestIsolationGuard.mark_test_observed()
inside a per-relationship function would lock out every OTHER candidate's
VALIDATE-phase evaluation the moment the FIRST candidate reached TEST -
clearly wrong (it would make "how many candidates get a fair VALIDATE-phase
evaluation" depend on iteration order). Instead this module runs in two
explicit passes over the WHOLE candidate set: evaluate_relationship_validate_phase
(DISCOVERED..ECONOMICALLY_SUPPORTED, calling assert_parameter_selection_allowed
before building each StrategyAgent/decision process - a real VALIDATE-phase
selection action) is run for every candidate FIRST; only once that full set
is frozen does the run call mark_test_observed ONCE and then
evaluate_pending_test_segment for exactly the candidates that reached
ECONOMICALLY_SUPPORTED, reusing their already-built, frozen agent/decision
process - no new parameter selection happens in this second pass. See
build_stage7_final_report for where the phase boundary actually sits.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from market_agent.experiment.portfolio_metrics import TRANSACTION_COST_PER_TRADE
from market_agent.learn.hierarchical_research import HierarchicalResearchReport, LevelTestResult
from market_agent.learn.hypothesis_testing import MIN_N, _matching_prior_rows
from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.strategy.decision_process import MethodologyDecisionProcess, build_validated_decision_process
from market_agent.strategy.outcome_engine import StrategyOutcomeReport, compute_strategy_outcome_report, compute_trade_outcome
from market_agent.strategy.strategy_agent import StrategyAgent
from market_agent.strategy.strategy_diagnostics import bootstrap_confidence_interval
from market_agent.strategy.test_isolation import TestIsolationGuard

EVIDENCE_STATES: tuple[str, ...] = (
    "DISCOVERED", "STATISTICALLY_SUPPORTED", "INCREMENTAL", "SHADOW", "ECONOMICALLY_SUPPORTED", "TEST_VALIDATED",
)

# Same minimum-sample discipline as every other statistical gate in this project (learn/hypothesis_testing.py's
# MIN_N) - not a separately tunable threshold invented for this report.
MIN_ECONOMIC_TRADES = MIN_N

FAR_FUTURE_ISO = "9999-12-31T00:00:00+00:00"  # sentinel upper bound for an "everything up to now" query


@dataclass
class RelationshipTrajectory:
    label: str
    concept: str
    condition: dict
    horizon_days: int
    reached_state: str
    rejection_reason: str | None
    relationship_id: str | None
    level_test_result: LevelTestResult
    methodology_ids: list[str] = field(default_factory=list)
    validate_outcome: StrategyOutcomeReport | None = None
    validate_bootstrap_ci: tuple[float, float] | None = None
    test_outcome: StrategyOutcomeReport | None = None
    test_bootstrap_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label, "concept": self.concept, "condition": self.condition,
            "horizon_days": self.horizon_days, "reached_state": self.reached_state,
            "rejection_reason": self.rejection_reason, "relationship_id": self.relationship_id,
            "methodology_ids": self.methodology_ids,
            "statistical_effect": self.level_test_result.test_result.mean_effect,
            "statistical_n": self.level_test_result.test_result.n,
            "validate_expectancy": self.validate_outcome.expectancy if self.validate_outcome else None,
            "validate_n_trades": self.validate_outcome.n_trades if self.validate_outcome else None,
            "validate_bootstrap_ci": self.validate_bootstrap_ci,
            "test_expectancy": self.test_outcome.expectancy if self.test_outcome else None,
            "test_n_trades": self.test_outcome.n_trades if self.test_outcome else None,
            "test_bootstrap_ci": self.test_bootstrap_ci,
        }


@dataclass
class PendingTestEvaluation:
    """A relationship that reached ECONOMICALLY_SUPPORTED in the VALIDATE
    phase, carrying the ALREADY-BUILT, frozen `decision_process`/`agent`
    forward so the TEST phase reuses them verbatim rather than
    reconstructing anything from TEST-segment information."""
    trajectory: RelationshipTrajectory
    decision_process: MethodologyDecisionProcess
    agent: StrategyAgent
    condition: dict
    horizon_days: int


@dataclass
class SegmentRunResult:
    """`abstain_reasons` (stage 7 item 9 clarity) buckets every ABSTAIN by
    its FIRST clause (e.g. "95% CI [...] spans zero..." collapses to one
    bucket regardless of the specific numbers) plus one bucket for a
    decision that committed to LONG/SHORT but had no usable OHLCV path -
    so a candidate with zero trades reports WHY, not just the bare count.
    This changes only what gets DISPLAYED about an already-made decision;
    no threshold or classification depends on it."""
    report: StrategyOutcomeReport | None
    bootstrap_ci: tuple[float, float] | None
    n_considered: int
    abstain_reasons: dict[str, int] = field(default_factory=dict)


def _run_segment(conn: sqlite3.Connection, ohlcv: OHLCVProvider, agent: StrategyAgent,
                  decision_process: MethodologyDecisionProcess, rows: list[sqlite3.Row], horizon_days: int,
                  transaction_cost: float) -> SegmentRunResult:
    trades, predicted_impacts, n_considered = [], [], 0
    abstain_reasons: dict[str, int] = {}
    for r in rows:
        n_considered += 1
        context = json.loads(r["context_json"])
        decision = agent.decide(decision_process, r["entity"], context)
        if decision.action not in ("LONG", "SHORT"):
            key = (decision.abstain_reason or "abstained, no reason recorded").split(" - ")[0][:80]
            abstain_reasons[key] = abstain_reasons.get(key, 0) + 1
            continue
        outcome = compute_trade_outcome(ohlcv, r["entity"], decision.action, datetime.fromisoformat(r["published_at"]),
                                          horizon_days, decision.invalidation_level, decision.confidence,
                                          context.get("regime"))
        if outcome is not None:
            trades.append(outcome)
            predicted_impacts.append(decision.predicted_return)
        else:
            key = "committed to a trade but no usable OHLCV bar path was found"
            abstain_reasons[key] = abstain_reasons.get(key, 0) + 1
    if not trades:
        return SegmentRunResult(None, None, n_considered, abstain_reasons)
    report = compute_strategy_outcome_report(trades, n_considered, predicted_impacts, transaction_cost)
    ci = bootstrap_confidence_interval([t.realized_return - transaction_cost for t in trades])
    return SegmentRunResult(report, ci, n_considered, abstain_reasons)


def _no_trades_reason(segment_label: str, result: SegmentRunResult) -> str:
    if result.n_considered == 0:
        return f"No real {segment_label}-segment observations matched this condition at all."
    top = sorted(result.abstain_reasons.items(), key=lambda kv: -kv[1])
    breakdown = "; ".join(f"{count}x {reason!r}" for reason, count in top[:3])
    return (f"0 real {segment_label}-segment trades out of {result.n_considered} decisions considered - "
            f"{breakdown}.")


def _find_relationship_row(conn: sqlite3.Connection, condition: dict, horizon_days: int) -> sqlite3.Row | None:
    for candidate in conn.execute("SELECT * FROM validated_relationships WHERE horizon_days = ?",
                                   (horizon_days,)).fetchall():
        if json.loads(candidate["condition_json"]) == condition:
            return candidate
    return None


def evaluate_relationship_validate_phase(conn: sqlite3.Connection, ohlcv: OHLCVProvider, label: str, concept: str,
                                          level_result: LevelTestResult, horizon_days: int,
                                          unconditional_baseline: dict[int, float], test_boundary: str,
                                          guard: TestIsolationGuard,
                                          transaction_cost: float = TRANSACTION_COST_PER_TRADE
                                          ) -> tuple[RelationshipTrajectory, PendingTestEvaluation | None]:
    """DISCOVERED through ECONOMICALLY_SUPPORTED (or an earlier rejection)
    for ONE candidate, using ONLY VALIDATE-segment (published before
    `test_boundary`) data. Returns (trajectory, pending) - `pending` is
    None unless the candidate reached ECONOMICALLY_SUPPORTED, in which
    case it carries the frozen decision_process/agent forward for the
    SEPARATE TEST-phase pass (evaluate_pending_test_segment) - see module
    docstring for why TEST evaluation is never done inline here."""
    condition = level_result.condition
    tr = RelationshipTrajectory(label=label, concept=concept, condition=condition, horizon_days=horizon_days,
                                 reached_state="DISCOVERED", rejection_reason=None, relationship_id=None,
                                 level_test_result=level_result)

    if level_result.test_result.status != "CONFIRMED":
        tr.rejection_reason = f"Failed the baseline-scalar significance test: {level_result.test_result.status}."
        return tr, None
    tr.reached_state = "STATISTICALLY_SUPPORTED"

    iv = level_result.incremental_value
    if iv is None or iv.status != "INCREMENTAL_VALUE_CONFIRMED":
        tr.rejection_reason = (f"Incremental-value check: {iv.status if iv else 'no diagnostic available'} - "
                                "does not beat CURRENT_ADAPTIVE's own prediction for the same real events.")
        return tr, None
    tr.reached_state = "INCREMENTAL"

    pt, ts = level_result.permutation_test, level_result.temporal_stability
    if pt is None or pt.status != "SURVIVES_PERMUTATION":
        tr.rejection_reason = (f"Permutation test: {pt.status if pt else 'unavailable'} - not distinguishable "
                                "from a random same-size subset of the same event population.")
        return tr, None
    if ts is None or ts.status != "STABLE_ACROSS_TIME":
        tr.rejection_reason = (f"Temporal stability: {ts.status if ts else 'unavailable'} - sign is not "
                                "consistent across the condition's own chronological halves.")
        return tr, None
    tr.reached_state = "SHADOW"  # the SAME status apply_test_results already assigned this relationship

    row = _find_relationship_row(conn, condition, horizon_days)
    if row is None:
        tr.rejection_reason = ("No validated_relationships row found for this condition - unexpected, since "
                                "SHADOW entry should have created one via apply_test_results.")
        return tr, None
    tr.relationship_id = row["relationship_id"]
    tr.methodology_ids = json.loads(row["methodology_ids_json"]) if row["methodology_ids_json"] else []

    guard.assert_parameter_selection_allowed(f"building a StrategyAgent decision process for relationship "
                                              f"{tr.relationship_id}")
    decision_process = build_validated_decision_process(conn, row, unconditional_baseline)
    if decision_process is None:
        tr.rejection_reason = ("No technical-concept-bearing setup in this relationship's own condition - "
                                "cannot build an executable decision process from event-context alone.")
        return tr, None

    agent = StrategyAgent(transaction_cost=transaction_cost)
    validate_rows = _matching_prior_rows(conn, condition, horizon_days, test_boundary)
    validate_run = _run_segment(conn, ohlcv, agent, decision_process, validate_rows, horizon_days, transaction_cost)
    validate_report, validate_ci = validate_run.report, validate_run.bootstrap_ci
    tr.validate_outcome, tr.validate_bootstrap_ci = validate_report, validate_ci

    if validate_report is None or validate_report.n_trades < MIN_ECONOMIC_TRADES:
        if validate_report is None:
            tr.rejection_reason = _no_trades_reason("VALIDATE", validate_run)
        else:
            tr.rejection_reason = (f"Only {validate_report.n_trades} real VALIDATE-segment trades - below "
                                    f"MIN_ECONOMIC_TRADES={MIN_ECONOMIC_TRADES}.")
        return tr, None

    economically_supported = (validate_report.expectancy is not None and validate_report.expectancy > 0
                               and validate_ci is not None and validate_ci[0] > 0
                               and (validate_report.profit_factor is None or validate_report.profit_factor > 1.0))
    if not economically_supported:
        tr.rejection_reason = (f"VALIDATE-segment economics do not clear the bar: expectancy="
                                f"{validate_report.expectancy!r}, bootstrap 95% CI={validate_ci!r}, "
                                f"profit_factor={validate_report.profit_factor!r}.")
        return tr, None
    tr.reached_state = "ECONOMICALLY_SUPPORTED"

    pending = PendingTestEvaluation(trajectory=tr, decision_process=decision_process, agent=agent,
                                     condition=condition, horizon_days=horizon_days)
    return tr, pending


def evaluate_pending_test_segment(conn: sqlite3.Connection, ohlcv: OHLCVProvider, pending: PendingTestEvaluation,
                                   test_boundary: str, transaction_cost: float = TRANSACTION_COST_PER_TRADE) -> None:
    """TEST phase for ONE already-ECONOMICALLY_SUPPORTED candidate - reuses
    `pending.agent`/`pending.decision_process` EXACTLY as built during the
    VALIDATE phase; nothing here selects or changes a parameter. Caller is
    responsible for having called guard.mark_test_observed() before this
    runs for ANY candidate in the batch - see build_stage7_final_report."""
    test_rows_all = _matching_prior_rows(conn, pending.condition, pending.horizon_days, FAR_FUTURE_ISO)
    test_rows = [r for r in test_rows_all if r["published_at"] >= test_boundary]
    test_run = _run_segment(conn, ohlcv, pending.agent, pending.decision_process, test_rows, pending.horizon_days,
                              transaction_cost)
    test_report, test_ci = test_run.report, test_run.bootstrap_ci
    tr = pending.trajectory
    tr.test_outcome, tr.test_bootstrap_ci = test_report, test_ci

    if test_report is None or test_report.n_trades < MIN_ECONOMIC_TRADES:
        if test_report is None:
            tr.rejection_reason = _no_trades_reason("TEST", test_run) + " Cannot assess TEST validation."
        else:
            tr.rejection_reason = (f"Only {test_report.n_trades} real TEST-segment trades - below "
                                    f"MIN_ECONOMIC_TRADES={MIN_ECONOMIC_TRADES}; cannot assess TEST validation.")
        return

    test_validated = (test_report.expectancy is not None and test_report.expectancy > 0
                       and test_ci is not None and test_ci[0] > 0)
    if not test_validated:
        tr.rejection_reason = (f"TEST-segment economics do not confirm: expectancy={test_report.expectancy!r}, "
                                f"bootstrap 95% CI={test_ci!r}.")
        return
    tr.reached_state = "TEST_VALIDATED"


def evaluate_relationship_trajectory(conn: sqlite3.Connection, ohlcv: OHLCVProvider, label: str, concept: str,
                                      level_result: LevelTestResult, horizon_days: int,
                                      unconditional_baseline: dict[int, float], test_boundary: str,
                                      guard: TestIsolationGuard,
                                      transaction_cost: float = TRANSACTION_COST_PER_TRADE) -> RelationshipTrajectory:
    """Convenience wrapper for evaluating a SINGLE relationship end-to-end
    (both phases back to back) - composes evaluate_relationship_validate_phase
    and evaluate_pending_test_segment. Fine to use standalone (e.g. in
    tests, or a one-off lookup) since there is only one candidate for the
    guard to lock out; a multi-candidate run MUST use the two functions
    directly with a proper phase boundary - see build_stage7_final_report
    and the module docstring for why."""
    tr, pending = evaluate_relationship_validate_phase(conn, ohlcv, label, concept, level_result, horizon_days,
                                                          unconditional_baseline, test_boundary, guard,
                                                          transaction_cost)
    if pending is not None:
        guard.mark_test_observed(f"evaluating relationship {tr.relationship_id} on the frozen TEST segment")
        evaluate_pending_test_segment(conn, ohlcv, pending, test_boundary, transaction_cost)
    return tr


def collect_trajectories_from_research_report(conn: sqlite3.Connection, ohlcv: OHLCVProvider,
                                                research: HierarchicalResearchReport,
                                                unconditional_baseline: dict[int, float], test_boundary: str,
                                                guard: TestIsolationGuard,
                                                transaction_cost: float = TRANSACTION_COST_PER_TRADE
                                                ) -> tuple[list[RelationshipTrajectory], list[PendingTestEvaluation]]:
    """VALIDATE-phase pass over every Level 2 and Level 3 candidate a
    single hierarchical research pass produced. Level 1 is deliberately
    excluded - it is a pure screening gate over pooled, non-default
    technical states and never produces a promotable relationship of its
    own (see learn/hierarchical_research.py's module docstring). Returns
    (all trajectories, pending TEST-phase evaluations) - the caller runs
    the TEST phase separately, after every research report's VALIDATE
    phase across the whole run is done."""
    concept_by_dimension = {fr.dimension: fr.concept for fr in research.level1_results}
    trajectories: list[RelationshipTrajectory] = []
    pending_evaluations: list[PendingTestEvaluation] = []

    for dimension, results in research.level2_results.items():
        concept = concept_by_dimension.get(dimension, dimension)
        for result in results:
            label = (f"L2 {research.event_type}/{research.direction}/{research.horizon_days}D "
                      f"{dimension}={result.condition.get(dimension)!r}")
            tr, pending = evaluate_relationship_validate_phase(
                conn, ohlcv, label, concept, result, research.horizon_days, unconditional_baseline, test_boundary,
                guard, transaction_cost)
            trajectories.append(tr)
            if pending is not None:
                pending_evaluations.append(pending)

    for setup_key, results in research.level3_results.items():
        dimension = setup_key.split("=", 1)[0]
        concept = concept_by_dimension.get(dimension, dimension)
        for result in results:
            extra = {k: v for k, v in result.condition.items() if k not in ("event_type", "direction", dimension)}
            label = f"L3 {research.event_type}/{research.direction}/{research.horizon_days}D {setup_key} + {extra}"
            tr, pending = evaluate_relationship_validate_phase(
                conn, ohlcv, label, concept, result, research.horizon_days, unconditional_baseline, test_boundary,
                guard, transaction_cost)
            trajectories.append(tr)
            if pending is not None:
                pending_evaluations.append(pending)

    return trajectories, pending_evaluations


def _state_counts(trajectories: list[RelationshipTrajectory]) -> dict[str, int]:
    counts = {s: 0 for s in EVIDENCE_STATES}
    for t in trajectories:
        counts[t.reached_state] += 1
    return counts


def answer_final_report_questions(trajectories: list[RelationshipTrajectory],
                                   five_way_summary: dict | None = None) -> dict[str, str]:
    """Answers the 8 questions stage 7 item 9 requires, computed directly
    from `trajectories` - never LLM-narrated, never fabricated. Each
    answer names the surviving relationships/methodologies explicitly, or
    says plainly that none survived - both are valid, reportable
    outcomes."""
    counts = _state_counts(trajectories)
    shadow_or_better = [t for t in trajectories if t.reached_state in
                         ("SHADOW", "ECONOMICALLY_SUPPORTED", "TEST_VALIDATED")]
    incremental_or_better = [t for t in trajectories if t.reached_state not in
                              ("DISCOVERED", "STATISTICALLY_SUPPORTED")]
    economically = [t for t in trajectories if t.reached_state in ("ECONOMICALLY_SUPPORTED", "TEST_VALIDATED")]
    test_validated = [t for t in trajectories if t.reached_state == "TEST_VALIDATED"]
    methodology_backed = sorted({mid for t in trajectories if t.reached_state != "DISCOVERED"
                                  for mid in t.methodology_ids})

    answers: dict[str, str] = {
        "what_does_the_system_know": (
            f"{len(trajectories)} candidate technical-concept relationships were evaluated end-to-end through "
            "the full evidence hierarchy (candidate -> statistical evidence -> incremental value -> temporal "
            "stability -> economic value -> TEST). State counts: "
            + ", ".join(f"{s}={counts[s]}" for s in EVIDENCE_STATES) + "."),
        "which_relationships_survived_all_controls": (
            ", ".join(t.label for t in shadow_or_better) if shadow_or_better else
            "None - no relationship survived the statistical-significance, incremental-value, permutation, and "
            "temporal-stability controls together."),
        "which_provide_incremental_information": (
            ", ".join(t.label for t in incremental_or_better) if incremental_or_better else
            "None - every relationship that cleared the baseline-scalar significance test added nothing beyond "
            "CURRENT_ADAPTIVE's own existing prediction for the same real events."),
        "which_methodologies_produced_useful_primitives": (
            ", ".join(methodology_backed) if methodology_backed else
            "None - no surviving relationship carries independent methodology provenance."),
        "which_primitives_form_executable_strategies": (
            ", ".join(t.label for t in economically) if economically else
            "None - no SHADOW-worthy relationship produced enough real VALIDATE-segment trades with positive, "
            "bootstrap-confirmed economics after transaction costs."),
        "do_strategies_produce_positive_economic_value_after_costs": (
            f"Yes, on VALIDATE-segment data: {len(economically)} relationship(s) "
            f"({', '.join(t.label for t in economically)})." if economically else
            "No relationship's strategy produced transaction-cost-adjusted positive expected value on VALIDATE "
            "data."),
        "does_adaptive_outperform_static": (
            five_way_summary.get("does_adaptive_outperform_static",
                                  "Five-way comparison summary supplied but has no answer for this question.")
            if five_way_summary is not None else
            "Not assessed by this report - pass a five_way_summary (from the five-way walk-forward comparison) "
            "to answer this question."),
        "does_any_advantage_survive_test": (
            f"Yes: {', '.join(t.label for t in test_validated)}." if test_validated else
            "No - no relationship's strategy retained positive, bootstrap-confirmed economic value on the "
            "frozen TEST segment. This is reported as a valid, honest scientific result, not a failure to hide."),
    }
    return answers


@dataclass
class Stage7FinalReport:
    generated_at: str
    test_boundary: str
    trajectories: list[RelationshipTrajectory]
    five_way_summary: dict | None
    answers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at, "test_boundary": self.test_boundary,
            "state_counts": _state_counts(self.trajectories),
            "trajectories": [t.to_dict() for t in self.trajectories],
            "five_way_summary": self.five_way_summary, "answers": self.answers,
        }

    def to_text(self) -> str:
        lines = [f"STAGE 7 FINAL REPORT - generated {self.generated_at}", f"TEST boundary: {self.test_boundary}",
                  "", "EVIDENCE-HIERARCHY STATE COUNTS:"]
        counts = _state_counts(self.trajectories)
        for s in EVIDENCE_STATES:
            lines.append(f"  {s}: {counts[s]}")
        lines.append("")
        lines.append("PER-CANDIDATE TRAJECTORIES:")
        for t in self.trajectories:
            lines.append(f"  [{t.reached_state}] {t.label}")
            if t.rejection_reason:
                lines.append(f"      -> {t.rejection_reason}")
            if t.validate_outcome is not None:
                lines.append(f"      VALIDATE: N={t.validate_outcome.n_trades} "
                              f"expectancy={t.validate_outcome.expectancy!r} CI={t.validate_bootstrap_ci!r}")
            if t.test_outcome is not None:
                lines.append(f"      TEST: N={t.test_outcome.n_trades} "
                              f"expectancy={t.test_outcome.expectancy!r} CI={t.test_bootstrap_ci!r}")
        lines.append("")
        lines.append("ANSWERS:")
        for question, answer in self.answers.items():
            lines.append(f"  Q: {question}")
            lines.append(f"  A: {answer}")
        return "\n".join(lines)


def build_stage7_final_report(conn: sqlite3.Connection, ohlcv: OHLCVProvider,
                               research_reports: list[HierarchicalResearchReport],
                               unconditional_baseline: dict[int, float], test_boundary: str,
                               guard: TestIsolationGuard | None = None, five_way_summary: dict | None = None,
                               transaction_cost: float = TRANSACTION_COST_PER_TRADE,
                               generated_at: str | None = None) -> Stage7FinalReport:
    """The single entry point: consumes already-computed
    HierarchicalResearchReport objects (one per event_type/direction/
    horizon combination a caller ran - see scripts/run_stage7_final_report.py
    for how the real run builds this list) and produces the complete,
    six-state-classified final report.

    PHASE BOUNDARY: every research report's VALIDATE phase runs FIRST,
    across the WHOLE candidate set - this is what freezes which
    relationships qualify for TEST evaluation using ONLY VALIDATE-segment
    information. Only then is guard.mark_test_observed() called ONCE for
    the whole run, and the TEST phase runs for exactly the frozen,
    ECONOMICALLY_SUPPORTED set. `guard` should be a single TestIsolationGuard
    if the caller wants to keep enforcing the boundary against later code
    in the same process; a fresh one is created if omitted."""
    guard = guard or TestIsolationGuard()
    trajectories: list[RelationshipTrajectory] = []
    pending_evaluations: list[PendingTestEvaluation] = []
    for research in research_reports:
        t, p = collect_trajectories_from_research_report(conn, ohlcv, research, unconditional_baseline,
                                                            test_boundary, guard, transaction_cost)
        trajectories.extend(t)
        pending_evaluations.extend(p)

    if pending_evaluations:
        guard.mark_test_observed(f"stage7_final_report: beginning frozen TEST-segment evaluation for "
                                  f"{len(pending_evaluations)} ECONOMICALLY_SUPPORTED candidate(s)")
    for pending in pending_evaluations:
        evaluate_pending_test_segment(conn, ohlcv, pending, test_boundary, transaction_cost)

    answers = answer_final_report_questions(trajectories, five_way_summary)
    return Stage7FinalReport(generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
                              test_boundary=test_boundary, trajectories=trajectories,
                              five_way_summary=five_way_summary, answers=answers)
