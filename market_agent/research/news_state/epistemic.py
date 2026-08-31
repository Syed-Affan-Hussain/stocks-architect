"""Refines extraction.py's FACT/REPORTING/INTERPRETATION/SPECULATION into
the five-way epistemic taxonomy this design needs: "management expects
20% growth" and "revenue grew 20%" must not carry the same evidentiary
weight, and extraction.py's existing REPORTING bucket does not distinguish
them (both currently just mean "someone said something," regardless of
who).

ADDS a new classifier ALONGSIDE the existing one - extraction.py's
classify_evidence_type is left completely alone (research/consistency.py
and research/risk.py already depend on its exact four-way output; breaking
it to add epistemic nuance would be exactly the "destroy working
functionality for a cleaner abstraction" this project explicitly avoids
everywhere else).

STILL KEYWORD-BASED, STILL DISCLOSED AS SUCH: this is a real, testable
refinement, not a claim of true source-attribution NLP. A clause that
genuinely quotes a named executive without any of the phrases below will
be under-classified as THIRD_PARTY_REPORTING (the same "default to the
weaker, more skeptical bucket when unsure" discipline extraction.py
already uses for its own REPORTING default).
"""
from __future__ import annotations

from market_agent.research.extraction import INTERPRETATION_CUES, SPECULATION_CUES

MANAGEMENT_ATTRIBUTION_CUES = (
    "management said", "management expects", "management stated", "management noted", "management raised",
    "management raises", "company said", "company expects", "company stated", "the company raised",
    "ceo said", "cfo said", "chief executive said", "chief financial officer said", "in a statement",
    "raises its guidance", "raised its guidance", "raises guidance", "raised guidance", "raises its outlook",
    "raised its outlook", "guidance to", "forecasts", "the company forecasts", "management forecasts",
)

# Ordered strongest -> weakest evidentiary weight - see schema.py's EPISTEMIC_STATUSES for the same
# order and CERTAINTY_WEIGHT below for the numeric mapping this order implies.
CERTAINTY_WEIGHT: dict[str, float] = {
    "OBSERVED_FACT": 1.0, "MANAGEMENT_CLAIM": 0.65, "THIRD_PARTY_REPORTING": 0.55,
    "ANALYST_INTERPRETATION": 0.4, "SPECULATION": 0.2,
}


def classify_epistemic_status(clause: str, source_type: str) -> str:
    """Priority order: SPECULATION first (a hedged forward-looking claim
    stays SPECULATION regardless of who is speaking - "management expects
    a possible decline" is still speculative content), then
    MANAGEMENT_CLAIM, then ANALYST_INTERPRETATION, then
    THIRD_PARTY_REPORTING as the default for news. SEC filings are always
    OBSERVED_FACT - primary company disclosure, same convention
    extraction.py's classify_evidence_type already uses."""
    if source_type == "SEC_FILING":
        return "OBSERVED_FACT"
    lower = clause.lower()
    if any(cue in lower for cue in SPECULATION_CUES):
        return "SPECULATION"
    if any(cue in lower for cue in MANAGEMENT_ATTRIBUTION_CUES):
        return "MANAGEMENT_CLAIM"
    if any(cue in lower for cue in INTERPRETATION_CUES):
        return "ANALYST_INTERPRETATION"
    return "THIRD_PARTY_REPORTING"


def certainty_for_breakdown(epistemic_breakdown: dict[str, int]) -> float:
    """[0,1] - a clause-count-weighted average of CERTAINTY_WEIGHT. A
    DISCLOSED HEURISTIC, not a calibrated probability - see llm_schema.py
    for why a real distribution (mu, sigma) isn't practical yet."""
    total = sum(epistemic_breakdown.values())
    if total == 0:
        return 0.0
    return sum(CERTAINTY_WEIGHT.get(status, 0.3) * count for status, count in epistemic_breakdown.items()) / total


def dominant_status(epistemic_breakdown: dict[str, int]) -> str:
    """The STRONGEST (not most frequent) status present - one real 10-Q
    disclosure among ten speculative headlines should not be described as
    'mostly speculative'. Ties within the same status are irrelevant since
    we return the status name, not a count."""
    if not epistemic_breakdown:
        return "THIRD_PARTY_REPORTING"
    present = [s for s, c in epistemic_breakdown.items() if c > 0]
    if not present:
        return "THIRD_PARTY_REPORTING"
    order = ["OBSERVED_FACT", "MANAGEMENT_CLAIM", "THIRD_PARTY_REPORTING", "ANALYST_INTERPRETATION", "SPECULATION"]
    return min(present, key=order.index)
