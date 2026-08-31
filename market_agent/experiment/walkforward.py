"""The chronological Static-vs-Adaptive walk-forward harness - Blueprint
sections P/Q, and stage 4's multi-horizon / shadow-deployment extensions.

CORE CORRECTNESS ARGUMENT (read this before modifying the loop below):
this harness processes real historical events as a single, strictly
time-ordered pass. At every point in that pass, the database contains
ONLY information whose trigger time (a prediction's `published_at`, or an
outcome's `resolve_at = published_at + horizon + embargo`) is <= the
event currently being handled. Concretely: for every new event, this
harness ALWAYS resolves any earlier-logged prediction whose embargoed
outcome window has closed (§ "outcome-horizon embargo" below) BEFORE
generating a new prediction for the new event - never after. This is what
makes "Agent-ADAPTIVE may only use information that would genuinely have
existed at that timestamp" true by construction rather than by a filter
that could be gotten wrong in one call site and not another. A dedicated
test (tests/test_walkforward.py::test_no_relationship_used_before_its_own_promotion_time)
audits this invariant directly rather than trusting the argument alone.

CONTEXT IS BUILT AT PREDICTION TIME, NOT UPFRONT - a stage-4 fix. Event
INTERPRETATION (deriving event_type/direction from raw text) is a pure
function of the text alone and is done in one upfront pass with no
leakage risk either way. CONTEXT (entity-history fields like "days since
this entity's last event", or "how many other entities had an event the
same day") depends on what's already been logged to episodic_events - if
built in that same upfront pass, before anything is logged, those fields
would always read empty/zero for every single event. Context is now
rebuilt fresh, from the live `conn`, immediately before each prediction
inside the main loop below.

MULTIPLE HORIZONS ARE STATISTICALLY INDEPENDENT - each horizon in
`config.horizon_days_list` gets its own burn-in baseline, its own
episodic_events rows, its own pending/resolve/embargo tracking, and its
own hypothesis tests. Nothing here derives one horizon's prediction from
another's.

OUTCOME-HORIZON EMBARGO: an event's outcome is not treated as "known" the
instant its raw horizon elapses (`published_at + horizon_days`) - an
extra `embargo_days` buffer is required first (`resolve_at = published_at
+ horizon_days + embargo_days`), preventing overlapping-label leakage.

SHADOW DEPLOYMENT: a confirmed hypothesis enters SHADOW, not ACTIVE - see
learn/shadow.py. This harness evaluates shadow relationships every time
new outcomes resolve, using the SAME chronological `now`.

FINAL HOLDOUT: Agent-ADAPTIVE keeps learning chronologically through the
ENTIRE dataset, including the reserved final-holdout segment - pausing
learning there would misrepresent what a genuinely deployed system does.
What "touched exactly once" actually constrains is METHODOLOGY: this
configuration (MIN_N, ALPHA, embargo, regime thresholds, shadow
probation size - all fixed constants elsewhere, none of them parameters
of this function) was fixed before this harness was run against real
data, and the final-holdout segment's reported metrics are not to be
used to go back and retune those constants, then re-run. This module
cannot enforce that discipline in code - it is a commitment about how the
RESULTS of a single run are used, recorded in the run's own report text.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from market_agent.events.interpret import Interpreter
from market_agent.events.schema import EventRecord
from market_agent.experiment.context import build_context
from market_agent.learn.error_taxonomy import classify_error
from market_agent.learn.governance import apply_test_results
from market_agent.learn.hypothesis import HypothesisGenerator, formalize_and_store
from market_agent.learn.hypothesis_testing import test_hypotheses_batch
from market_agent.learn.revalidation import run_revalidation_pass
from market_agent.learn.shadow import evaluate_shadow_relationships
from market_agent.llm.select import describe_active_choice
from market_agent.outcomes.observe import PriceSeriesProvider, compute_abnormal_return
from market_agent.pipeline import EMPTY_CONTEXT_PLACEHOLDER
from market_agent.sources.edgar_guidance import SourcedRawItem
from market_agent.store import db

VARIANT_LABEL = "v2_regime_and_prior_return_conditioned_hypothesis"  # this run's ONE tracked variant -
#   v1 (regime-only) was tested against real data in stage 3 and did not outperform STATIC on the
#   final holdout; v2 adds prior-return-bucket conditioning (see learn/hypothesis.py) as a SEPARATE,
#   independently-motivated hypothesis about the mechanism (directly requested by this stage's own
#   context-expansion item), not a retune of v1 to force a different answer - see module docstring.


@dataclass
class ScoredPrediction:
    event_id: str
    entity: str
    published_at: datetime
    horizon_days: int
    agent: str  # "STATIC" | "ADAPTIVE"
    predicted_impact: float | None
    predicted_confidence: str
    basis: dict
    realized_abnormal_return: float | None
    error_type: str | None
    segment: str  # "DEVELOPMENT" | "FINAL_HOLDOUT"
    generalization_case: bool  # ADAPTIVE only: True if the relationship used was learned from a DIFFERENT entity


@dataclass
class WalkforwardConfig:
    horizon_days_list: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    embargo_days: int = 2
    benchmark_ticker: str = "SPY"
    burn_in_fraction: float = 0.20
    final_holdout_fraction: float = 0.20


@dataclass
class WalkforwardReport:
    variant_label: str
    interpreter_used: str
    hypothesis_generator_used: str
    n_raw_items: int
    n_interpreted: int
    n_burn_in: int
    horizon_days_list: list[int]
    unconditional_baseline: dict  # {horizon_days: magnitude}
    n_development: int = 0
    n_final_holdout: int = 0
    scored: list[ScoredPrediction] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    revalidations: list[dict] = field(default_factory=list)
    shadow_evaluations: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def _pending_entry(static_id, adaptive_id, event: EventRecord, horizon_days: int, resolve_at, segment):
    return {"static_id": static_id, "adaptive_id": adaptive_id, "event": event, "horizon_days": horizon_days,
            "resolve_at": resolve_at, "segment": segment}


def run_walkforward(sourced_items: list[SourcedRawItem], prices: PriceSeriesProvider, interpreter: Interpreter,
                     hypothesis_generator: HypothesisGenerator, config: WalkforwardConfig,
                     conn: sqlite3.Connection | None = None) -> WalkforwardReport:
    conn = conn or db.connect(":memory:")
    items = sorted(sourced_items, key=lambda i: i.raw_item.published_at)

    evidence = [describe_active_choice(interpreter, hypothesis_generator),
                f"Variant: {VARIANT_LABEL}",
                f"{len(items)} raw items, chronological range "
                f"{items[0].raw_item.published_at.date() if items else 'n/a'} to "
                f"{items[-1].raw_item.published_at.date() if items else 'n/a'}.",
                f"Horizons (statistically independent): {config.horizon_days_list}"]

    # --- interpret every raw item up front - pure function of its own text, no leakage risk.
    #     Context is deliberately NOT built here - see module docstring - a placeholder is attached
    #     and overwritten at prediction time in the main loop below. ---
    interpreted: list[EventRecord] = []
    for sourced in items:
        event = interpreter.interpret(sourced.raw_item, EMPTY_CONTEXT_PLACEHOLDER, source_reliability_snapshot=None)
        if event is not None:
            interpreted.append(event)
    evidence.append(f"{len(interpreted)}/{len(items)} raw items interpreted as in-scope events.")

    n_burn_in = int(len(interpreted) * config.burn_in_fraction)
    n_holdout_start = len(interpreted) - int(len(interpreted) * config.final_holdout_fraction)
    evidence.append(f"Burn-in: first {n_burn_in} interpreted events (baseline estimation only, not scored). "
                     f"Final holdout: last {len(interpreted) - n_holdout_start} interpreted events, reserved, "
                     "touched once.")

    burn_in_events, remaining = interpreted[:n_burn_in], interpreted[n_burn_in:]
    unconditional_baseline = _estimate_baseline(prices, burn_in_events, config)
    for h, mag in unconditional_baseline.items():
        evidence.append(f"Unconditional baseline at {h}d: {mag:+.2%} (unsigned magnitude).")

    from market_agent.agents.adaptive_agent import AdaptiveAgent
    from market_agent.agents.static_agent import StaticAgent
    static = StaticAgent(unconditional_baseline, model_version="STATIC_v1")
    adaptive = AdaptiveAgent(conn, unconditional_baseline, model_version="ADAPTIVE_v1")

    report = WalkforwardReport(VARIANT_LABEL, interpreter.NAME, hypothesis_generator.NAME, len(items),
                                len(interpreted), len(burn_in_events), config.horizon_days_list,
                                unconditional_baseline, evidence=evidence)

    pending: list[dict] = []
    last_revalidated_quarter: tuple[int, int] | None = None
    n_dev_count = n_holdout_start - n_burn_in

    for idx, event in enumerate(remaining):
        now = event.published_at
        pending = _resolve_due(conn, pending, now, prices, config, hypothesis_generator, report)
        last_revalidated_quarter = _maybe_revalidate(conn, now, last_revalidated_quarter,
                                                       unconditional_baseline, report)

        # Context built fresh, from the live conn, right before predicting - see module docstring.
        event.context = build_context(prices, event.entity, event.published_at, config.benchmark_ticker,
                                       event.event_type, conn).to_dict()
        segment = "DEVELOPMENT" if idx < n_dev_count else "FINAL_HOLDOUT"

        for horizon_days in config.horizon_days_list:
            knowledge_version = db.count_governance_changes(conn)
            static_pred = static.predict(event, horizon_days, now)
            adaptive_pred = adaptive.predict(event, horizon_days, now)
            static_pred.knowledge_version = knowledge_version
            adaptive_pred.knowledge_version = knowledge_version
            static_id = db.log_prediction(conn, event, static_pred)
            adaptive_id = db.log_prediction(conn, event, adaptive_pred)
            resolve_at = now + timedelta(days=horizon_days + config.embargo_days)
            pending.append(_pending_entry(static_id, adaptive_id, event, horizon_days, resolve_at, segment))

    if pending:
        pending = _resolve_due(conn, pending, pending[-1]["resolve_at"], prices, config, hypothesis_generator,
                                report, force=True)

    report.n_development = sum(1 for s in report.scored if s.segment == "DEVELOPMENT" and s.agent == "ADAPTIVE")
    report.n_final_holdout = sum(1 for s in report.scored if s.segment == "FINAL_HOLDOUT" and s.agent == "ADAPTIVE")
    return report


def _estimate_baseline(prices: PriceSeriesProvider, burn_in_events: list[EventRecord],
                        config: WalkforwardConfig) -> dict[int, float]:
    baseline = {}
    for horizon_days in config.horizon_days_list:
        magnitudes = []
        for event in burn_in_events:
            result = compute_abnormal_return(prices, event.entity, config.benchmark_ticker, event.published_at,
                                              horizon_days)
            if result.status == "OK":
                magnitudes.append(abs(result.abnormal_return))
        baseline[horizon_days] = (sum(magnitudes) / len(magnitudes)) if magnitudes else 0.02
    return baseline


def _resolve_due(conn, pending, now, prices, config, hypothesis_generator, report: WalkforwardReport,
                  force: bool = False):
    due = [p for p in pending if force or p["resolve_at"] <= now]
    still_pending = [p for p in pending if not (force or p["resolve_at"] <= now)]
    if not due:
        return still_pending

    newly_flagged_hypotheses = []
    for entry in due:
        event, static_id, adaptive_id = entry["event"], entry["static_id"], entry["adaptive_id"]
        horizon_days, segment = entry["horizon_days"], entry["segment"]
        result = compute_abnormal_return(prices, event.entity, config.benchmark_ticker, event.published_at,
                                          horizon_days)
        for agent_name, event_id in (("STATIC", static_id), ("ADAPTIVE", adaptive_id)):
            row = db.get_event(conn, event_id)
            error = classify_error(row["predicted_impact"], row["predicted_confidence"],
                                    result.abnormal_return, result.status)
            db.record_outcome(conn, event_id, result.abnormal_return if result.status == "OK" else None,
                               entry["resolve_at"], error.error_value, error.error_type)
            basis = json.loads(row["prediction_basis_json"] or "{}")
            generalization_case = False
            if agent_name == "ADAPTIVE" and basis.get("basis") == "validated_relationship":
                generalization_case = _is_generalization_case(conn, basis["relationship_id"], event.entity)
            report.scored.append(ScoredPrediction(
                event_id=event_id, entity=event.entity, published_at=event.published_at,
                horizon_days=horizon_days, agent=agent_name, predicted_impact=row["predicted_impact"],
                predicted_confidence=row["predicted_confidence"], basis=basis,
                realized_abnormal_return=result.abnormal_return, error_type=error.error_type,
                segment=segment, generalization_case=generalization_case,
            ))
            if agent_name == "ADAPTIVE" and error.may_learn_from:
                hids = formalize_and_store(conn, hypothesis_generator, row, error.error_type,
                                            horizon_days, proposed_at=entry["resolve_at"])
                newly_flagged_hypotheses.extend(hids if isinstance(hids, list) else ([hids] if hids else []))

    if newly_flagged_hypotheses:
        pending_rows = db.untested_hypotheses(conn)
        results = test_hypotheses_batch(conn, pending_rows, report.unconditional_baseline)
        apply_test_results(conn, results, promoted_by="walkforward-harness", clock_now=now)
        for r in results:
            entry_dict = {"hypothesis_id": r.hypothesis_id, "status": r.status, "n": r.n,
                          "mean_effect": r.mean_effect, "p_value": r.p_value, "p_value_corrected": r.p_value_corrected}
            (report.promotions if r.status == "CONFIRMED" else report.rejections).append(entry_dict)

    shadow_summary = evaluate_shadow_relationships(conn, report.unconditional_baseline,
                                                     promoted_by="walkforward-harness", clock_now=now)
    report.shadow_evaluations.extend(shadow_summary)

    return still_pending


def _is_generalization_case(conn, relationship_id: str, current_entity: str) -> bool:
    rel = conn.execute("SELECT source_hypothesis_id FROM validated_relationships WHERE relationship_id = ?",
                        (relationship_id,)).fetchone()
    if rel is None or rel["source_hypothesis_id"] is None:
        return False
    hyp = conn.execute("SELECT source_event_id FROM candidate_hypotheses WHERE hypothesis_id = ?",
                        (rel["source_hypothesis_id"],)).fetchone()
    if hyp is None:
        return False
    source_event = db.get_event(conn, hyp["source_event_id"])
    return source_event is not None and source_event["entity"] != current_entity


def _maybe_revalidate(conn, now, last_quarter, baseline, report: WalkforwardReport):
    current_quarter = (now.year, (now.month - 1) // 3 + 1)
    if current_quarter == last_quarter:
        return last_quarter
    summary = run_revalidation_pass(conn, baseline, promoted_by="walkforward-harness", clock_now=now)
    report.revalidations.extend(summary)
    return current_quarter
