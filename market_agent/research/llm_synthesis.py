"""Item 19: the LLM architecture for this product. Reuses market_agent's
EXISTING LLMClient/LLMNotConfiguredError contract (market_agent/llm/
interpreter.py) rather than defining a second, parallel one - same
no-silent-fallback discipline as market_agent/llm/select.py: no client
configured means the DETERMINISTIC path runs, and `llm_status` says so
explicitly in every report, never silently.

THE LLM NEVER INVENTS FINANCIAL FACTS (item 19's explicit requirement):
`synthesize_executive_summary` only ever hands the client ALREADY-COMPUTED
structured evidence (the same narratives/risks/assessment every other part
of this report is built from) and asks it to WRITE ABOUT that evidence via
`complete_structured` (schema-validated output, not free-form text) - it
is never asked to produce numbers, dates, or claims on its own.

NO LLM CLIENT IS CONFIGURED IN THIS ENVIRONMENT BY DEFAULT: the
deterministic template below is what actually runs unless a caller
constructs and passes a real LLMClient - it is a genuine, tested v1, not
a placeholder, but it is templated prose, not true semantic synthesis.
"""
from __future__ import annotations

from market_agent.llm.interpreter import LLMClient
from market_agent.research.schema import Narrative, Risk

SUMMARY_SCHEMA = {"type": "object", "properties": {"executive_summary": {"type": "string"}},
                   "required": ["executive_summary"]}


def llm_status(client: LLMClient | None) -> str:
    return "UNAVAILABLE" if client is None else f"ACTIVE:{type(client).__name__}"


def _deterministic_summary(entity: str, assessment: str, narratives: list[Narrative], risks: list[Risk]) -> str:
    if not narratives and not risks:
        return (f"No usable evidence was collected for {entity} in this research pass - the assessment is "
               "INSUFFICIENT_EVIDENCE. See the Sources section for which providers were reachable.")
    top_narrative = max(narratives, key=lambda n: n.independent_source_count, default=None)
    top_risk = next((r for r in risks if r.severity == "HIGH"), risks[0] if risks else None)
    parts = [f"Current research assessment for {entity}: {assessment.replace('_', ' ').title()}."]
    if top_narrative:
        parts.append(f"The most substantiated storyline in recent coverage is: {top_narrative.description}")
    if top_risk:
        parts.append(f"The most significant identified risk: {top_risk.description}")
    n_pos = sum(1 for n in narratives if n.sentiment == "POSITIVE")
    n_neg = sum(1 for n in narratives if n.sentiment == "NEGATIVE")
    parts.append(f"Overall, {n_pos} narrative(s) skew positive and {n_neg} skew negative across the evidence "
                 "collected in this pass.")
    return " ".join(parts)


def synthesize_executive_summary(client: LLMClient | None, entity: str, assessment: str,
                                  narratives: list[Narrative], risks: list[Risk]) -> str:
    if client is None:
        return _deterministic_summary(entity, assessment, narratives, risks)
    prompt = (
        f"Write a 3-4 sentence executive summary of the research evidence below for {entity}. "
        f"Assessment: {assessment}. Do NOT invent any fact, number, or date not present below.\n\n"
        f"Narratives: {[n.description for n in narratives]}\n"
        f"Risks: {[r.description for r in risks]}"
    )
    result = client.complete_structured(prompt, SUMMARY_SCHEMA)
    return result["executive_summary"]
