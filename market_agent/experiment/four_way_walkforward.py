"""The four-way chronological walk-forward harness - stage 6's extension
of walkforward.py's Static-vs-Adaptive comparison to STATIC / CURRENT
ADAPTIVE / TECHNICAL ADAPTIVE / METHODOLOGY-INFORMED ADAPTIVE (see
agents/variants.py for what distinguishes the three ADAPTIVE variants -
they share one AdaptiveAgent implementation and one governed
validated_relationships table, differing only in which relationships each
is eligible to match).

REUSES walkforward.py's CORE INVARIANTS RATHER THAN RE-DERIVING THEM: this
module imports `_estimate_baseline`, `_is_generalization_case`, and
`_maybe_revalidate` directly from experiment/walkforward.py instead of
duplicating them - they encode the SAME point-in-time/leakage-defense
logic that module's own module docstring documents at length, and nothing
about a fourth scored agent changes any of it. The main loop below follows
the identical resolve-before-predict ordering walkforward.py's module
docstring explains is what makes point-in-time correctness true "by
construction rather than by a filter."

ONE agent drives hypothesis generation, not four: CURRENT_ADAPTIVE plays
the exact role "ADAPTIVE" played in walkforward.py's two-agent version -
its prediction errors are what gets diagnosed into candidate hypotheses
(learn/hypothesis.py's generator itself pulls the FULL bounded pool of
event-context + technical dimensions from the event's own real context,
regardless of which agent's error triggered the call - see that module's
docstring - so this choice does not restrict which hypotheses ever get
proposed, only which agent's prediction is used to CLASSIFY the error that
triggers the proposal). TECHNICAL_ADAPTIVE and METHODOLOGY_ADAPTIVE are
purely READ-ONLY observers of the same governed relationships table.

FINAL HOLDOUT / TEST: two available disciplines, chosen per run via
`freeze_governance_during_test` -

  DEFAULT (False) - identical to walkforward.py's original design: all
    four agents keep predicting AND LEARNING chronologically through the
    entire dataset including the reserved final-holdout segment; what
    "touched once" constrains is METHODOLOGY (this run's fixed
    configuration), not literal code execution against that segment. This
    is a deliberate position (see walkforward.py's own docstring): a
    genuinely deployed system keeps learning as new data arrives, and
    artificially freezing it at a holdout boundary would misrepresent
    that.

STAGE 7: A FIFTH AGENT, ENSEMBLE_ADAPTIVE (agents/variants.py::
make_ensemble_adaptive_agent) - despite the module/function names kept
as "four_way" for backward compatibility (existing callers/tests are
unaffected), AGENT_NAMES now has five entries. ENSEMBLE_ADAPTIVE starts
with an EMPTY qualified-relationship set (behaves exactly like STATIC
through the whole VALIDATE segment - nothing has been verified yet), and
is reconfigured EXACTLY ONCE, at the VALIDATE/TEST boundary, via the
optional `compute_qualified_relationships_fn(conn, test_boundary_iso,
unconditional_baseline)` callback - called with the live `conn`, the
boundary's own ISO timestamp, and the SAME unconditional_baseline dict
this run already computed from burn-in data (so the callback can run
further point-in-time-correct queries, e.g. a fresh learn/
hierarchical_research.py pass pinned at that exact boundary using the
SAME baseline every other agent in this run uses, without having to
independently re-derive either value) at the moment chronological time
first reaches the TEST segment, so it only ever sees VALIDATE-and-earlier
data (the SAME boundary `freeze_governance_during_test` uses), satisfying
"learned only from information available at that chronological point." If
no callback is supplied (default), ENSEMBLE_ADAPTIVE stays baseline-only
for the whole run - a disclosed no-op, not silently broken.

  STAGE 7 STRICTER DISCIPLINE (True) - TRAIN -> VALIDATE -> SHADOW -> TEST,
    a real gap found while building this: with the default off, a
    relationship's CONFIRMATION, PROMOTION, and REVALIDATION can currently
    happen using outcomes published DURING the final-holdout window, and
    that governance action can then change what OTHER, later final-holdout
    predictions see (AdaptiveAgent always queries whatever is currently
    ACTIVE) - a real form of holdout contamination for the governance
    process itself, even though REPORTED metrics are still correctly
    split by segment. Setting this flag True freezes all governance
    (hypothesis proposal/testing, SHADOW->ACTIVE promotion, revalidation)
    the moment chronological time crosses into the final-holdout segment -
    predictions still happen normally using whatever was ACTIVE by then,
    just nothing more gets learned or promoted. TRAIN = the burn-in
    segment (baseline estimation only, unchanged); VALIDATE = the
    DEVELOPMENT segment (where all governance happens); SHADOW = the
    existing per-relationship probation window (learn/shadow.py,
    unchanged - it is not a separate fixed calendar segment, it is
    per-relationship and occurs WITHIN VALIDATE); TEST = the
    FINAL_HOLDOUT segment, now genuinely frozen for governance purposes.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from market_agent.events.interpret import Interpreter
from market_agent.events.schema import EventRecord
from market_agent.experiment.context import build_context
from market_agent.experiment.walkforward import _estimate_baseline, _is_generalization_case, _maybe_revalidate
from market_agent.learn.error_taxonomy import classify_error
from market_agent.learn.governance import apply_test_results
from market_agent.learn.hypothesis import HypothesisGenerator, formalize_and_store
from market_agent.learn.hypothesis_testing import test_hypotheses_batch
from market_agent.learn.shadow import evaluate_shadow_relationships
from market_agent.llm.select import describe_active_choice
from market_agent.outcomes.observe import PriceSeriesProvider, compute_abnormal_return
from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.pipeline import EMPTY_CONTEXT_PLACEHOLDER
from market_agent.sources.edgar_guidance import SourcedRawItem
from market_agent.store import db

VARIANT_LABEL = "v3_four_way_technical_and_methodology_conditioned"

AGENT_NAMES: tuple[str, ...] = ("STATIC", "CURRENT_ADAPTIVE", "TECHNICAL_ADAPTIVE", "METHODOLOGY_ADAPTIVE",
                                 "ENSEMBLE_ADAPTIVE")
DIAGNOSIS_AGENT = "CURRENT_ADAPTIVE"  # see module docstring - the one agent whose errors drive learning


@dataclass
class FourWayScoredPrediction:
    event_id: str
    entity: str
    published_at: datetime
    horizon_days: int
    agent: str  # one of AGENT_NAMES
    predicted_impact: float | None
    predicted_confidence: str
    basis: dict
    realized_abnormal_return: float | None
    error_type: str | None
    segment: str  # "DEVELOPMENT" | "FINAL_HOLDOUT"
    generalization_case: bool  # True if the relationship used was learned from a DIFFERENT entity


@dataclass
class FourWayWalkforwardConfig:
    horizon_days_list: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    embargo_days: int = 2
    benchmark_ticker: str = "SPY"
    burn_in_fraction: float = 0.20
    final_holdout_fraction: float = 0.20
    freeze_governance_during_test: bool = False  # see run_four_way_walkforward's module-level note
    #   below on TRAIN/VALIDATE/SHADOW/TEST - default False preserves this module's original,
    #   already-tested behavior (ADAPTIVE keeps learning through the whole dataset) for every
    #   existing caller; set True for the stricter stage-7 discipline that treats the final holdout
    #   as pure evaluation, never governance.


@dataclass
class FourWayWalkforwardReport:
    variant_label: str
    interpreter_used: str
    hypothesis_generator_used: str
    n_raw_items: int
    n_interpreted: int
    n_burn_in: int
    horizon_days_list: list[int]
    unconditional_baseline: dict
    n_development: int = 0
    n_final_holdout: int = 0
    scored: list[FourWayScoredPrediction] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    revalidations: list[dict] = field(default_factory=list)
    shadow_evaluations: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def _pending_entry(event_ids: dict[str, str], event: EventRecord, horizon_days: int, resolve_at, segment):
    return {"event_ids": event_ids, "event": event, "horizon_days": horizon_days,
            "resolve_at": resolve_at, "segment": segment}


def _make_agents(conn: sqlite3.Connection, baseline: dict[int, float]) -> dict[str, object]:
    from market_agent.agents.static_agent import StaticAgent
    from market_agent.agents.variants import (
        make_current_adaptive_agent, make_ensemble_adaptive_agent, make_methodology_informed_adaptive_agent,
        make_technical_adaptive_agent,
    )
    return {
        "STATIC": StaticAgent(baseline, model_version="STATIC_v1"),
        "CURRENT_ADAPTIVE": make_current_adaptive_agent(conn, baseline),
        "TECHNICAL_ADAPTIVE": make_technical_adaptive_agent(conn, baseline),
        "METHODOLOGY_ADAPTIVE": make_methodology_informed_adaptive_agent(conn, baseline),
        # starts with an EMPTY qualified set - see module docstring's "STAGE 7: A FIFTH AGENT" note
        # for when/how this gets reconfigured.
        "ENSEMBLE_ADAPTIVE": make_ensemble_adaptive_agent(conn, baseline, qualified_relationship_ids=set()),
    }


def run_four_way_walkforward(sourced_items: list[SourcedRawItem], prices: PriceSeriesProvider,
                              interpreter: Interpreter, hypothesis_generator: HypothesisGenerator,
                              config: FourWayWalkforwardConfig, conn: sqlite3.Connection | None = None,
                              ohlcv: OHLCVProvider | None = None,
                              compute_qualified_relationships_fn=None) -> FourWayWalkforwardReport:
    conn = conn or db.connect(":memory:")
    items = sorted(sourced_items, key=lambda i: i.raw_item.published_at)

    evidence = [describe_active_choice(interpreter, hypothesis_generator),
                f"Variant: {VARIANT_LABEL}",
                f"Agents compared: {AGENT_NAMES}",
                f"{len(items)} raw items, chronological range "
                f"{items[0].raw_item.published_at.date() if items else 'n/a'} to "
                f"{items[-1].raw_item.published_at.date() if items else 'n/a'}.",
                f"Horizons (statistically independent): {config.horizon_days_list}"]

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

    agents = _make_agents(conn, unconditional_baseline)

    report = FourWayWalkforwardReport(VARIANT_LABEL, interpreter.NAME, hypothesis_generator.NAME, len(items),
                                       len(interpreted), len(burn_in_events), config.horizon_days_list,
                                       unconditional_baseline, evidence=evidence)

    pending: list[dict] = []
    last_revalidated_quarter: tuple[int, int] | None = None
    n_dev_count = n_holdout_start - n_burn_in
    # TEST boundary for freeze_governance_during_test AND for the ENSEMBLE_ADAPTIVE reconfiguration
    # below - the published_at of the first FINAL_HOLDOUT event, i.e. the moment chronological time
    # crosses VALIDATE -> TEST. None if there's no holdout.
    test_boundary = remaining[n_dev_count].published_at if n_dev_count < len(remaining) else None
    ensemble_reconfigured = False

    for idx, event in enumerate(remaining):
        now = event.published_at
        governance_frozen = (config.freeze_governance_during_test and test_boundary is not None
                              and now >= test_boundary)
        pending = _resolve_due(conn, pending, now, prices, config, hypothesis_generator, report,
                                governance_frozen=governance_frozen)
        if not governance_frozen:
            last_revalidated_quarter = _maybe_revalidate(conn, now, last_revalidated_quarter,
                                                           unconditional_baseline, report)

        if (compute_qualified_relationships_fn is not None and not ensemble_reconfigured
                and test_boundary is not None and now >= test_boundary):
            # Exactly once, at the moment VALIDATE ends - see module docstring's "STAGE 7: A FIFTH
            # AGENT" note. `conn` at this instant contains only VALIDATE-and-earlier governance
            # actions (freeze_governance_during_test, if enabled, already guarantees nothing later
            # has been written yet either way, since this fires at the SAME boundary).
            qualified_ids = compute_qualified_relationships_fn(conn, test_boundary.isoformat(), unconditional_baseline)
            agents["ENSEMBLE_ADAPTIVE"].qualified_relationship_ids = qualified_ids
            report.evidence.append(f"ENSEMBLE_ADAPTIVE reconfigured at the VALIDATE/TEST boundary with "
                                    f"{len(qualified_ids)} qualified relationship(s).")
            ensemble_reconfigured = True

        event.context = build_context(prices, event.entity, event.published_at, config.benchmark_ticker,
                                       event.event_type, conn, ohlcv=ohlcv).to_dict()
        segment = "DEVELOPMENT" if idx < n_dev_count else "FINAL_HOLDOUT"

        for horizon_days in config.horizon_days_list:
            knowledge_version = db.count_governance_changes(conn)
            event_ids = {}
            for agent_name, agent in agents.items():
                pred = agent.predict(event, horizon_days, now)
                pred.knowledge_version = knowledge_version
                event_ids[agent_name] = db.log_prediction(conn, event, pred)
            resolve_at = now + timedelta(days=horizon_days + config.embargo_days)
            pending.append(_pending_entry(event_ids, event, horizon_days, resolve_at, segment))

    if pending:
        final_governance_frozen = config.freeze_governance_during_test and test_boundary is not None
        pending = _resolve_due(conn, pending, pending[-1]["resolve_at"], prices, config, hypothesis_generator,
                                report, force=True, governance_frozen=final_governance_frozen)

    report.n_development = sum(1 for s in report.scored
                                if s.segment == "DEVELOPMENT" and s.agent == DIAGNOSIS_AGENT)
    report.n_final_holdout = sum(1 for s in report.scored
                                  if s.segment == "FINAL_HOLDOUT" and s.agent == DIAGNOSIS_AGENT)
    return report


def _resolve_due(conn, pending, now, prices, config, hypothesis_generator, report: FourWayWalkforwardReport,
                  force: bool = False, governance_frozen: bool = False):
    """`governance_frozen` (stage 7's TRAIN/VALIDATE/SHADOW/TEST discipline
    - see module docstring) is a SINGLE flag computed once by the caller
    from the chronological clock, not a per-entry check: once TEST time
    is reached, no NEW hypothesis is formalized, no batch test/promotion
    runs, and no shadow evaluation runs, for the rest of this call -
    deliberately the more conservative of the two possible designs (a
    late-resolving VALIDATE-period prediction's outcome is also excluded
    once we're past the boundary, rather than trying to thread a
    per-source-segment exception through shadow evaluation's own "any new
    evidence since shadow_started_at" query, which doesn't distinguish
    which SEGMENT a piece of evidence was originally predicted in)."""
    due = [p for p in pending if force or p["resolve_at"] <= now]
    still_pending = [p for p in pending if not (force or p["resolve_at"] <= now)]
    if not due:
        return still_pending

    newly_flagged_hypotheses = []
    for entry in due:
        event, event_ids = entry["event"], entry["event_ids"]
        horizon_days, segment = entry["horizon_days"], entry["segment"]
        result = compute_abnormal_return(prices, event.entity, config.benchmark_ticker, event.published_at,
                                          horizon_days)
        for agent_name in AGENT_NAMES:
            event_id = event_ids[agent_name]
            row = db.get_event(conn, event_id)
            error = classify_error(row["predicted_impact"], row["predicted_confidence"],
                                    result.abnormal_return, result.status)
            db.record_outcome(conn, event_id, result.abnormal_return if result.status == "OK" else None,
                               entry["resolve_at"], error.error_value, error.error_type)
            basis = json.loads(row["prediction_basis_json"] or "{}")
            generalization_case = False
            if basis.get("basis") == "validated_relationship":
                generalization_case = _is_generalization_case(conn, basis["relationship_id"], event.entity)
            report.scored.append(FourWayScoredPrediction(
                event_id=event_id, entity=event.entity, published_at=event.published_at,
                horizon_days=horizon_days, agent=agent_name, predicted_impact=row["predicted_impact"],
                predicted_confidence=row["predicted_confidence"], basis=basis,
                realized_abnormal_return=result.abnormal_return, error_type=error.error_type,
                segment=segment, generalization_case=generalization_case,
            ))
            if agent_name == DIAGNOSIS_AGENT and error.may_learn_from and not governance_frozen:
                hids = formalize_and_store(conn, hypothesis_generator, row, error.error_type,
                                            horizon_days, proposed_at=entry["resolve_at"])
                newly_flagged_hypotheses.extend(hids)

    if governance_frozen:
        return still_pending  # TEST segment - predictions/scoring above still happened; no governance below

    if newly_flagged_hypotheses:
        pending_rows = db.untested_hypotheses(conn)
        results = test_hypotheses_batch(conn, pending_rows, report.unconditional_baseline)
        apply_test_results(conn, results, promoted_by="four-way-walkforward-harness", clock_now=now)
        for r in results:
            entry_dict = {"hypothesis_id": r.hypothesis_id, "status": r.status, "n": r.n,
                          "mean_effect": r.mean_effect, "p_value": r.p_value, "p_value_corrected": r.p_value_corrected}
            (report.promotions if r.status == "CONFIRMED" else report.rejections).append(entry_dict)

    shadow_summary = evaluate_shadow_relationships(conn, report.unconditional_baseline,
                                                     promoted_by="four-way-walkforward-harness", clock_now=now)
    report.shadow_evaluations.extend(shadow_summary)

    return still_pending
