"""The reusable single-event prediction pipeline (item 1) and the
market-level query API (item 12). Nothing here is a new prediction
mechanism - it assembles already-existing, already-tested components
(interpret -> context -> retrieve -> predict -> package) into one
reusable path, so the same logic serves the walk-forward harness's bulk
replay AND a live, on-demand "what does the system think about this
security right now" query without duplicating the assembly logic.

Event interpretation itself stays a pure function of RawItem text (see
events/interpret.py) - designed around an Interpreter/HypothesisGenerator
INTERFACE from stage 3 onward specifically so a new event type (earnings,
analyst actions, other filings) plugs in as a new interpreter
implementation without touching anything in this module, agents/, or
learn/ - those all operate on the already-interpreted EventRecord, never
on raw text.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from market_agent.agents.base import PredictionAgent
from market_agent.events.interpret import Interpreter
from market_agent.events.schema import ContextSnapshot, EventRecord, PredictionRecord, RawItem
from market_agent.experiment.context import build_context
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.pit.clock import PointInTimeClock
from market_agent.retrieval.similarity import find_similar_cases

EMPTY_CONTEXT_PLACEHOLDER = ContextSnapshot(regime="UNKNOWN", prior_5d_return=None, sector_momentum="UNKNOWN")

# Below this many retrieved similar historical cases, a prediction that fell back to the
# unconditional baseline (no validated relationship backing it) is downgraded to
# INSUFFICIENT_PRECEDENT rather than reported with confidence - item 12's explicit "do not force a
# prediction when the evidence does not support one". A prediction backed by a validated
# relationship is NOT subject to this override - that relationship's own n_supporting already is
# its evidence of precedent, established through the full hypothesis-testing gate, independent of
# how many cases this specific bucketed retrieval query happens to surface.
MIN_SIMILAR_CASES_FOR_UNCONDITIONAL_CONFIDENCE = 1


@dataclass
class SecurityPrediction:
    entity: str
    event_type: str
    horizon_days: int
    status: str                       # "OK" | "INSUFFICIENT_PRECEDENT"
    predicted_impact: float | None
    predicted_confidence: str
    uncertainty: float | None
    basis: dict
    regime: str
    novelty_score: float              # 0 (many precedents) .. 1 (none found)
    n_similar_cases: int
    similar_case_ids: list[str]
    reasoning_provenance: list[str] = field(default_factory=list)


def interpret_event(interpreter: Interpreter, raw_item: RawItem, context: ContextSnapshot,
                     clock: PointInTimeClock) -> EventRecord | None:
    """Item 1's first pipeline step, as a standalone reusable call -
    point-in-time checked (raises if raw_item is timestamped after the
    clock's current time) before interpretation is even attempted."""
    clock.assert_not_future(raw_item.published_at, label=f"{raw_item.source} item for {raw_item.entity}")
    return interpreter.interpret(raw_item, context, source_reliability_snapshot=None)


def predict_for_security(conn: sqlite3.Connection, agent: PredictionAgent, event: EventRecord,
                          horizon_days_list: list[int], prices: PriceSeriesProvider,
                          predicted_at: datetime) -> list[SecurityPrediction]:
    """Item 12's market-level API: 'what is the expected impact of the
    current event over each horizon' - with confidence, uncertainty,
    historical evidence, similar cases, the active relationship (if any),
    regime, novelty, and reasoning provenance, or an explicit
    INSUFFICIENT_PRECEDENT per horizon where the evidence doesn't support
    a confident number. Does NOT log anything to episodic_events itself -
    that's log_prediction's job (store/db.py), called separately by
    whatever orchestrates a real prediction (the walk-forward harness, or
    a live caller) once it decides this query result should become a
    permanent record."""
    context = event.context
    results = []
    for horizon_days in horizon_days_list:
        prediction = agent.predict(event, horizon_days, predicted_at)
        similar = find_similar_cases(conn, event.event_type, context.get("regime", "UNKNOWN"),
                                      context.get("prior_5d_return"), horizon_days,
                                      published_before=predicted_at.isoformat(), outcome_known_only=True)
        novelty_score = 1.0 / (1 + len(similar))

        status = "OK"
        confidence = prediction.predicted_confidence
        impact = prediction.predicted_impact
        uncertainty = None
        reasoning = [f"Regime: {context.get('regime', 'UNKNOWN')}.",
                     f"Retrieved {len(similar)} similar historical case(s) (same event type, regime, and "
                     f"prior-return bucket, resolved before this prediction)."]

        if prediction.basis.get("basis") == "validated_relationship":
            rel_id = prediction.basis.get("relationship_id")
            n_supp = prediction.basis.get("n_supporting")
            reasoning.append(f"Using validated relationship {rel_id} (N={n_supp} supporting observations).")
            rel_row = conn.execute("SELECT ci_low, ci_high FROM validated_relationships WHERE relationship_id = ?",
                                    (rel_id,)).fetchone()
            if rel_row is not None and rel_row["ci_low"] is not None and rel_row["ci_high"] is not None:
                uncertainty = (rel_row["ci_high"] - rel_row["ci_low"]) / 2.0
        elif prediction.predicted_confidence == "INSUFFICIENT_PRECEDENT" or impact is None:
            status = "INSUFFICIENT_PRECEDENT"
            reasoning.append("No baseline available for this horizon - abstaining rather than guessing.")
        elif len(similar) < MIN_SIMILAR_CASES_FOR_UNCONDITIONAL_CONFIDENCE:
            status = "INSUFFICIENT_PRECEDENT"
            confidence = "INSUFFICIENT_PRECEDENT"
            reasoning.append(f"Falling back to the unconditional baseline, but {len(similar)} directly comparable "
                              "precedent(s) were found - insufficient evidence to report this with confidence "
                              "rather than as an unconditioned population average.")
        else:
            reasoning.append("Using the unconditional baseline (no validated relationship matches this context).")

        results.append(SecurityPrediction(
            entity=event.entity, event_type=event.event_type, horizon_days=horizon_days, status=status,
            predicted_impact=impact if status == "OK" else impact, predicted_confidence=confidence,
            uncertainty=uncertainty, basis=prediction.basis, regime=context.get("regime", "UNKNOWN"),
            novelty_score=novelty_score, n_similar_cases=len(similar),
            similar_case_ids=[c.event_id for c in similar[:10]], reasoning_provenance=reasoning,
        ))
    return results


@dataclass
class EventPredictionResult:
    """Item 10's predict_event(event) return shape - "NOT_RECOGNIZED" (the
    interpreter found no in-scope event in this text) or "OK" with the
    interpreted event and its per-horizon predictions attached."""
    status: str            # "NOT_RECOGNIZED" | "OK"
    entity: str
    raw_text: str
    event: EventRecord | None = None
    predictions: list[SecurityPrediction] = field(default_factory=list)


def predict_event(conn: sqlite3.Connection, agent: PredictionAgent, prices: PriceSeriesProvider,
                   interpreter: Interpreter, entity: str, raw_text: str, source: str, published_at: datetime,
                   predicted_at: datetime, horizon_days_list: list[int],
                   benchmark_ticker: str = "SPY") -> EventPredictionResult:
    """Item 10's predict_event(event) API: composes interpret -> build
    real context -> predict_for_security into one reusable call, the same
    two-pass pattern experiment/walkforward.py and market_agent/predict.py
    already use (interpret against a placeholder context first - the
    interpreter never inspects context content, only raw text - so a
    price-derived context is never computed for text that isn't even an
    in-scope event).

    Deliberately does NOT log anything to episodic_events - same
    "query vs record" split predict_for_security itself documents. A
    caller that wants this result to become a permanent ledger row must
    call store.db.log_prediction itself (market_agent/predict.py's CLI
    does exactly that); a read-only "what would the system predict right
    now" query (e.g. from a portfolio dashboard, item 11) should never
    have a ledger write forced onto it."""
    clock = PointInTimeClock(now=predicted_at)
    raw_item = RawItem(text=raw_text, source=source, entity=entity, published_at=published_at)
    event = interpret_event(interpreter, raw_item, EMPTY_CONTEXT_PLACEHOLDER, clock)
    if event is None:
        return EventPredictionResult(status="NOT_RECOGNIZED", entity=entity, raw_text=raw_text)

    context = build_context(prices, entity, published_at, benchmark_ticker, event.event_type, conn)
    event.context = context.to_dict()
    predictions = predict_for_security(conn, agent, event, horizon_days_list, prices, predicted_at)
    return EventPredictionResult(status="OK", entity=entity, raw_text=raw_text, event=event, predictions=predictions)


@dataclass
class RelevantRelationship:
    relationship_id: str
    event_type: str
    direction: str
    horizon_days: int
    effect_estimate: float
    ci_low: float | None
    ci_high: float | None
    n_supporting: int
    condition: dict


@dataclass
class SecurityOutlook:
    entity: str
    as_of: str
    regime: str
    prior_5d_return: float | None
    realized_vol_20d: float | None
    applicable_relationships: list[RelevantRelationship] = field(default_factory=list)
    recent_predictions: list[dict] = field(default_factory=list)
    reasoning_provenance: list[str] = field(default_factory=list)


def _condition_matches_context(condition: dict, context: dict) -> bool:
    """Same matching rule as agents/adaptive_agent.py's private helper of
    the same purpose - duplicated rather than imported across that
    module's boundary (it's five lines, and agents/ shouldn't need to
    export prediction-matching internals just for this query path)."""
    for key, value in condition.items():
        if key in ("event_type", "direction"):
            continue
        if context.get(key) != value:
            return False
    return True


def predict_security(conn: sqlite3.Connection, entity: str, prices: PriceSeriesProvider, as_of: datetime,
                      benchmark_ticker: str = "SPY", recent_limit: int = 5) -> SecurityOutlook:
    """Item 10's predict_security(security, context) API. Answers what the
    system currently knows that's RELEVANT to this security - NOT a price
    target, and NOT a forecast of an event that hasn't happened. Three
    specific questions: (1) what regime/volatility context is this
    security in right now, (2) which ACTIVE validated relationships (across
    every event type this system currently understands, not just one)
    would apply IF a matching event occurred right now - explicitly
    conditional, (3) what has this system actually predicted for this
    entity recently, from the permanent ledger. This is the read the
    portfolio-translation layer (item 11) queries per holding."""
    context = build_context(prices, entity, as_of, benchmark_ticker, event_type="ANY", conn=None)
    context_dict = context.to_dict()

    applicable = []
    for row in conn.execute("SELECT * FROM validated_relationships WHERE status = 'ACTIVE'").fetchall():
        condition = json.loads(row["condition_json"])
        if _condition_matches_context(condition, context_dict):
            applicable.append(RelevantRelationship(
                relationship_id=row["relationship_id"], event_type=condition.get("event_type", "UNKNOWN"),
                direction=condition.get("direction", "UNKNOWN"), horizon_days=row["horizon_days"],
                effect_estimate=row["effect_estimate"], ci_low=row["ci_low"], ci_high=row["ci_high"],
                n_supporting=row["n_supporting"], condition=condition,
            ))

    recent_rows = conn.execute(
        "SELECT * FROM episodic_events WHERE entity = ? ORDER BY predicted_at DESC LIMIT ?",
        (entity, recent_limit)).fetchall()
    recent_predictions = [
        {"event_id": r["event_id"], "event_type": r["event_type"], "direction": r["direction"],
         "horizon_days": r["horizon_days"], "predicted_impact": r["predicted_impact"],
         "predicted_confidence": r["predicted_confidence"], "predicted_at": r["predicted_at"],
         "realized_abnormal_return": r["realized_abnormal_return"], "outcome_locked": bool(r["outcome_locked"])}
        for r in recent_rows]

    reasoning = [f"Regime: {context.regime}.",
                 f"{len(applicable)} ACTIVE relationship(s) would apply if a matching event occurred right "
                 "now (conditional - no such event has been observed).",
                 f"{len(recent_predictions)} prior prediction(s) on record for this entity."]

    return SecurityOutlook(entity=entity, as_of=as_of.isoformat(), regime=context.regime,
                            prior_5d_return=context.prior_5d_return, realized_vol_20d=context.realized_vol_20d,
                            applicable_relationships=applicable, recent_predictions=recent_predictions,
                            reasoning_provenance=reasoning)
