"""Item 15: structured risk extraction, derived from evidence this
pipeline ALREADY collected - never a separate, independently-invented
risk list. Every Risk traces back to specific narratives/contradictions/
fundamental facts via `evidence_source_ids`.
"""
from __future__ import annotations

import hashlib

from market_agent.research.schema import Contradiction, FundamentalFact, Narrative, Risk, TimelineEvent

AREA_TO_RISK_CATEGORY: dict[str, str] = {
    "regulatory": "REGULATORY", "geopolitical": "GEOPOLITICAL", "competition": "COMPETITION",
    "supply_chain": "SUPPLY_CHAIN", "workforce": "EXECUTION", "management": "EXECUTION",
    "debt": "BALANCE_SHEET", "capital_allocation": "BALANCE_SHEET", "cash_flow": "LIQUIDITY",
    "demand": "BUSINESS", "revenue": "BUSINESS", "margins": "BUSINESS", "earnings": "BUSINESS",
    "guidance": "BUSINESS", "product": "BUSINESS",
}

# Fixed, disclosed heuristic - NOT investment advice, just a simple, transparent leverage check.
DEBT_TO_CASH_CONCERN_RATIO = 3.0


def _severity_for(narrative: Narrative) -> str:
    if narrative.trend in ("STRENGTHENING", "EMERGING") and narrative.independent_source_count >= 2:
        return "HIGH"
    if narrative.independent_source_count == 1:
        return "LOW"
    return "MEDIUM"


def _status_for(narrative: Narrative, events_by_id: dict[str, TimelineEvent]) -> str:
    ids = narrative.supporting_event_ids + narrative.contradicting_event_ids
    evidence_types = [events_by_id[i].evidence_type for i in ids if i in events_by_id]
    if not evidence_types:
        return "EMERGING"
    if all(t == "SPECULATION" for t in evidence_types):
        return "SPECULATIVE"
    if narrative.trend == "EMERGING":
        return "EMERGING"
    return "KNOWN"


def extract_risks_from_narratives(narratives: list[Narrative], events_by_id: dict[str, TimelineEvent]) -> list[Risk]:
    risks: list[Risk] = []
    for n in narratives:
        if n.sentiment != "NEGATIVE":
            continue
        category = AREA_TO_RISK_CATEGORY.get(n.affected_area, "BUSINESS")
        source_ids = [sid for eid in (n.supporting_event_ids + n.contradicting_event_ids)
                      for sid in events_by_id[eid].source_ids if eid in events_by_id]
        risk_id = "R_" + hashlib.sha256(n.narrative_id.encode()).hexdigest()[:12]
        risks.append(Risk(
            risk_id=risk_id, category=category, status=_status_for(n, events_by_id),
            description=n.description, severity=_severity_for(n), confidence=n.confidence,
            evidence_source_ids=source_ids, recent_change=n.trend, affected_area=n.affected_area,
        ))
    return risks


def extract_risks_from_contradictions(contradictions: list[Contradiction]) -> list[Risk]:
    risks = []
    for c in contradictions:
        risk_id = "R_" + hashlib.sha256(("contradiction:" + c.contradiction_id).encode()).hexdigest()[:12]
        risks.append(Risk(
            risk_id=risk_id, category="NARRATIVE", status="EMERGING",
            description=f"Unresolved conflicting evidence: {c.description}", severity="MEDIUM",
            confidence="LOW", evidence_source_ids=c.side_a_source_ids + c.side_b_source_ids,
            recent_change="DISPUTED", affected_area=None,
        ))
    return risks


def extract_risks_from_fundamentals(fundamentals: list[FundamentalFact], source_id: str | None) -> list[Risk]:
    """A small, fixed, disclosed set of balance-sheet/liquidity heuristics
    computed directly from the SAME real disclosed facts - never a
    separate, undisclosed judgment."""
    by_tag = {f.tag: f for f in fundamentals}
    risks: list[Risk] = []
    debt, cash = by_tag.get("LongTermDebt"), by_tag.get("CashAndCashEquivalents")
    if debt and cash and debt.value is not None and cash.value is not None and cash.value > 0:
        ratio = debt.value / cash.value
        if ratio > DEBT_TO_CASH_CONCERN_RATIO:
            risks.append(Risk(
                risk_id="R_balance_sheet_leverage", category="BALANCE_SHEET", status="KNOWN",
                description=f"Long-term debt (${debt.value:,.0f}) is {ratio:.1f}x disclosed cash and equivalents "
                            f"(${cash.value:,.0f}) as of {debt.period_end}.",
                severity="MEDIUM" if ratio < DEBT_TO_CASH_CONCERN_RATIO * 2 else "HIGH", confidence="HIGH",
                evidence_source_ids=[source_id] if source_id else [], recent_change=None,
                affected_area="debt",
            ))
    fcf = by_tag.get("FreeCashFlow")
    if fcf and fcf.value is not None and fcf.value < 0:
        risks.append(Risk(
            risk_id="R_negative_fcf", category="LIQUIDITY", status="KNOWN",
            description=f"Free cash flow was negative (${fcf.value:,.0f}) for the period ended {fcf.period_end}.",
            severity="HIGH", confidence="HIGH", evidence_source_ids=[source_id] if source_id else [],
            recent_change=None, affected_area="cash_flow",
        ))
    return risks


def extract_all_risks(narratives: list[Narrative], events_by_id: dict[str, TimelineEvent],
                       contradictions: list[Contradiction], fundamentals: list[FundamentalFact],
                       fundamentals_source_id: str | None) -> list[Risk]:
    risks = (extract_risks_from_narratives(narratives, events_by_id)
             + extract_risks_from_contradictions(contradictions)
             + extract_risks_from_fundamentals(fundamentals, fundamentals_source_id))
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risks.sort(key=lambda r: severity_rank.get(r.severity, 3))
    return risks
