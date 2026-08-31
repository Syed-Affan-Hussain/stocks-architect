"""Portfolio translation layer - stage 5 item 11: event -> security impact
-> exposure/factor propagation -> portfolio impact.

PORTFOLIO COMPOSITION IS NEVER A TRAINING TARGET, ENFORCED BY WHAT THIS
MODULE DOESN'T DO: nothing here writes to episodic_events,
validated_relationships, candidate_hypotheses, or model_registry. This is
a pure, read-only DOWNSTREAM CONSUMER of pipeline.py's already-tested
predict_event()/predict_security() - it takes a portfolio (just a plain
{entity: weight} mapping, never persisted, never influences what the
system learns) and reports what the system's EXISTING knowledge implies
for it. A portfolio full of one investor's specific holdings must never
change what the system believes is true about the market - that would be
overfitting the system's beliefs to one user's exposure, exactly the
failure mode the blueprint's category separation exists to prevent.

DISCLOSED LIMITATION - NO CROSS-SECURITY PROPAGATION MODEL: "factor
propagation" here means exactly one thing - a holding OTHER than the
triggering entity is reported with its OWN current regime and its OWN
applicable ACTIVE relationships (via predict_security), never a derived
spillover computed from the triggering event through some assumed
correlation/beta/sector-exposure model. No such model exists in this
system (ContextSnapshot's own docstring already discloses the same gap:
no sector classification, no cross-security correlation data source).
Reporting a fabricated spillover number would be worse than reporting
none - so `portfolio_expected_impact` only ever aggregates the triggering
entity's own predicted impact, weighted by its portfolio weight; every
other holding's contribution is explicitly None, not a guessed value.

NO "BEATS THE MARKET" FRAMING: nothing in this module's output should be
read as investment advice or a claim of predictive edge - see the
project's standing constraint (stage 5 item 12) against that framing.
This reports what a specific, already-validated (or not) relationship
implies, with its own confidence/uncertainty attached, nothing more.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.pipeline import EventPredictionResult, predict_security


@dataclass
class HoldingImpact:
    entity: str
    weight: float
    is_directly_affected: bool          # True only for the entity the triggering event is actually about
    status: str                         # "DIRECT_EVENT" | "INSUFFICIENT_PRECEDENT" | "NO_DIRECT_EVENT"
    predicted_impact: float | None      # the entity's own predicted impact at the requested horizon
    predicted_confidence: str | None
    weighted_contribution: float | None  # weight * predicted_impact - None wherever predicted_impact is None
    regime: str
    n_applicable_relationships: int     # from predict_security - what the system currently knows about this
    #                                     holding right now, independent of the triggering event


@dataclass
class PortfolioImpactReport:
    as_of: str
    triggering_entity: str
    triggering_event_type: str | None
    horizon_days: int
    holdings: list[HoldingImpact] = field(default_factory=list)
    portfolio_expected_impact: float | None = None  # None unless the triggering entity is held AND has a
    #                                                   real predicted_impact at this horizon - never partially
    #                                                   estimated from other holdings.
    n_holdings_with_data: int = 0
    reasoning_provenance: list[str] = field(default_factory=list)


def translate_event_to_portfolio(conn: sqlite3.Connection, prices: PriceSeriesProvider,
                                  portfolio: dict[str, float], event_prediction: EventPredictionResult,
                                  horizon_days: int, as_of: datetime) -> PortfolioImpactReport:
    """`portfolio` is a plain {entity: weight} mapping the caller owns -
    this function never stores it anywhere. `event_prediction` is
    whatever pipeline.predict_event() already returned (this module does
    not call predict_event itself, so a caller who wants a live 'what if'
    read composes them explicitly - keeping the two concerns separate:
    'what would the system predict' vs 'what does that mean for THIS
    portfolio')."""
    if event_prediction.status != "OK":
        return PortfolioImpactReport(
            as_of=as_of.isoformat(), triggering_entity=event_prediction.entity, triggering_event_type=None,
            horizon_days=horizon_days,
            reasoning_provenance=[f"No in-scope event recognized for {event_prediction.entity} - "
                                   "nothing to translate to portfolio impact."])

    matching = next((p for p in event_prediction.predictions if p.horizon_days == horizon_days), None)

    holdings: list[HoldingImpact] = []
    portfolio_expected_impact = 0.0
    n_with_data = 0
    for entity, weight in portfolio.items():
        is_direct = entity == event_prediction.entity
        outlook = predict_security(conn, entity, prices, as_of)

        if is_direct and matching is not None and matching.status == "OK" and matching.predicted_impact is not None:
            contribution = weight * matching.predicted_impact
            portfolio_expected_impact += contribution
            n_with_data += 1
            holdings.append(HoldingImpact(
                entity=entity, weight=weight, is_directly_affected=True, status="DIRECT_EVENT",
                predicted_impact=matching.predicted_impact, predicted_confidence=matching.predicted_confidence,
                weighted_contribution=contribution, regime=outlook.regime,
                n_applicable_relationships=len(outlook.applicable_relationships)))
        elif is_direct:
            holdings.append(HoldingImpact(
                entity=entity, weight=weight, is_directly_affected=True, status="INSUFFICIENT_PRECEDENT",
                predicted_impact=None, predicted_confidence=None, weighted_contribution=None,
                regime=outlook.regime, n_applicable_relationships=len(outlook.applicable_relationships)))
        else:
            holdings.append(HoldingImpact(
                entity=entity, weight=weight, is_directly_affected=False, status="NO_DIRECT_EVENT",
                predicted_impact=None, predicted_confidence=None, weighted_contribution=None,
                regime=outlook.regime, n_applicable_relationships=len(outlook.applicable_relationships)))

    n_other_holdings = len(portfolio) - (1 if event_prediction.entity in portfolio else 0)
    reasoning = [
        f"Triggering event: {event_prediction.event.entity} {event_prediction.event.event_type} "
        f"({event_prediction.event.direction}).",
        "Only the triggering entity's own predicted impact is propagated into portfolio_expected_impact - "
        "this system has no cross-security correlation/beta model (disclosed gap, not fabricated).",
        f"{n_other_holdings} other holding(s) reported with their OWN current regime and applicable ACTIVE "
        "relationship count only - not a derived spillover from this event.",
    ]
    if event_prediction.entity not in portfolio:
        reasoning.append(f"{event_prediction.entity} is not a current holding - portfolio_expected_impact is None.")

    return PortfolioImpactReport(
        as_of=as_of.isoformat(), triggering_entity=event_prediction.entity,
        triggering_event_type=event_prediction.event.event_type, horizon_days=horizon_days, holdings=holdings,
        portfolio_expected_impact=(portfolio_expected_impact if n_with_data > 0 else None),
        n_holdings_with_data=n_with_data, reasoning_provenance=reasoning,
    )
