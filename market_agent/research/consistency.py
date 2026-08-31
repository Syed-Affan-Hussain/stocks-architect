"""Items 10/14: does a narrative match what the company actually
disclosed, and where does the evidence explicitly conflict?

HONEST ABOUT WHAT FUNDAMENTALS CAN AND CANNOT CONFIRM: only narratives
about performance-adjacent topics (revenue, earnings, guidance, demand,
margins, cash flow) are checked against fundamentals at all - a narrative
about management changes, regulatory developments, or competition is
correctly reported INSUFFICIENT_EVIDENCE here, never forced into
SUPPORTED/CONTRADICTED using data that has nothing to do with it.

CONTRADICTIONS ARE FOUND, NOT INVENTED: a Contradiction is only raised
when a SINGLE narrative genuinely contains BOTH supporting and
contradicting events (narratives.py already separates these) - i.e. real,
opposing claims about the SAME topic exist in the collected evidence, not
a manufactured disagreement between unrelated narratives.
"""
from __future__ import annotations

import hashlib

from market_agent.research.fundamentals import explain_fundamentals
from market_agent.research.schema import Contradiction, ConsistencyCheck, Narrative, TimelineEvent

PERFORMANCE_AREAS = {"revenue", "earnings", "guidance", "demand", "margins", "cash_flow"}


def _fundamental_direction(facts_json: dict | None) -> tuple[str, list[str]]:
    """POSITIVE/NEGATIVE/MIXED/UNKNOWN, derived from the SAME real,
    data-driven sentences fundamentals.py already produces - never a
    second, independent judgment of the numbers."""
    if facts_json is None:
        return "UNKNOWN", []
    lines = explain_fundamentals(facts_json)
    pos = sum(1 for l in lines if "grew" in l)
    neg = sum(1 for l in lines if "declined" in l)
    if pos == 0 and neg == 0:
        return "UNKNOWN", lines
    if pos and neg:
        return "MIXED", lines
    return ("POSITIVE" if pos else "NEGATIVE"), lines


def check_consistency(narratives: list[Narrative], facts_json: dict | None) -> list[ConsistencyCheck]:
    fundamental_direction, evidence_lines = _fundamental_direction(facts_json)
    checks: list[ConsistencyCheck] = []
    for n in narratives:
        if n.affected_area not in PERFORMANCE_AREAS:
            checks.append(ConsistencyCheck(
                narrative_id=n.narrative_id, verdict="INSUFFICIENT_EVIDENCE",
                explanation=f"Disclosed fundamentals do not directly speak to '{n.affected_area or 'this topic'}' "
                            "- consistency cannot be assessed from financial statements alone."))
            continue
        if fundamental_direction == "UNKNOWN":
            checks.append(ConsistencyCheck(
                narrative_id=n.narrative_id, verdict="INSUFFICIENT_EVIDENCE",
                explanation="No usable year-over-year fundamental comparison is available to check this "
                            "narrative against."))
            continue

        narrative_signal = n.sentiment
        if narrative_signal == "NEUTRAL":
            checks.append(ConsistencyCheck(narrative_id=n.narrative_id, verdict="INSUFFICIENT_EVIDENCE",
                                            explanation="Narrative carries no clear directional claim to check."))
            continue

        if fundamental_direction == "MIXED" or narrative_signal == "MIXED":
            verdict = "PARTIALLY_SUPPORTED"
            explanation = ("Both the narrative's reporting and the disclosed fundamentals show a mixed picture - "
                           "neither clearly confirms nor clearly contradicts the other.")
        elif narrative_signal == fundamental_direction:
            verdict = "SUPPORTED"
            explanation = (f"The narrative's {narrative_signal.lower()} framing is consistent with the "
                           f"company's own disclosed {fundamental_direction.lower()} fundamental trend.")
        else:
            verdict = "CONTRADICTED"
            explanation = (f"The narrative's {narrative_signal.lower()} framing is NOT supported by the "
                           f"company's own disclosed fundamentals, which show a {fundamental_direction.lower()} "
                           "trend over the same period.")
        checks.append(ConsistencyCheck(narrative_id=n.narrative_id, verdict=verdict, explanation=explanation,
                                        supporting_fact_tags=["Revenues", "NetIncomeLoss"]))
    return checks


def detect_contradictions(narratives: list[Narrative], events_by_id: dict[str, TimelineEvent]) -> list[Contradiction]:
    contradictions: list[Contradiction] = []
    for n in narratives:
        if not n.supporting_event_ids or not n.contradicting_event_ids:
            continue
        support_event = events_by_id.get(n.supporting_event_ids[0])
        contradict_event = events_by_id.get(n.contradicting_event_ids[0])
        if support_event is None or contradict_event is None:
            continue
        contradiction_id = "C_" + hashlib.sha256(n.narrative_id.encode()).hexdigest()[:12]
        contradictions.append(Contradiction(
            contradiction_id=contradiction_id,
            description=f"Reporting on '{n.description}' is not unanimous - some sources frame this positively, "
                       "others negatively, within the same underlying topic.",
            side_a=support_event.description, side_a_source_ids=support_event.source_ids,
            side_b=contradict_event.description, side_b_source_ids=contradict_event.source_ids,
            what_would_resolve_it=("A primary-source disclosure (an SEC filing or direct company statement) "
                                   "addressing this topic directly would outweigh secondary reporting on either "
                                   "side."),
        ))
    return contradictions
