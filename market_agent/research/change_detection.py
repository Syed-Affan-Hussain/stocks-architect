"""Items 11/18: "what changed since the last research pass for this
company" - reuses the persistent research_reports ledger (store/db.py
schema v5) so `research NVDA` run a second time updates state rather than
starting from zero.

DETERMINISTIC IDS MAKE THIS A REAL DIFF, NOT A GUESS: narrative_id
(narratives.py) and risk_id (risk.py) are both derived deterministically
from (entity, event_type, affected_area) - the SAME underlying topic
produces the SAME id across two separate research passes. This lets
change detection be a plain set/dict comparison between the current
report and the prior one's stored JSON, not a fuzzy text-similarity guess.

NEVER CLAIMS CAUSALITY (item 11's explicit requirement): a sentiment or
assessment change is reported as a FACT about what the system's own
output changed to, never as "because of X" unless a specific new event
is the obvious, sole driver - and even then the wording stays descriptive
("new development observed"), not causal ("caused the change").
"""
from __future__ import annotations

from market_agent.research.schema import ChangeSummary, Narrative, Risk


def detect_changes(current_narratives: list[Narrative], current_risks: list[Risk], current_assessment: str,
                    prior_report_json: dict | None) -> ChangeSummary:
    if prior_report_json is None:
        return ChangeSummary(has_prior_report=False,
                              evidence=["This is the first research pass for this company - no prior report to "
                                        "compare against."])

    prior_narratives_by_id = {n["narrative_id"]: n for n in prior_report_json.get("narratives", [])}
    prior_risk_ids = {r["risk_id"] for r in prior_report_json.get("risks", [])}
    prior_assessment = prior_report_json.get("assessment")

    sentiment_changes = []
    narrative_changes = []
    for n in current_narratives:
        prior = prior_narratives_by_id.get(n.narrative_id)
        if prior is None:
            narrative_changes.append(f"NEW narrative: {n.description}")
            continue
        if prior.get("sentiment") != n.sentiment:
            sentiment_changes.append(f"{n.description}: {prior.get('sentiment')} -> {n.sentiment}")
        if prior.get("trend") != n.trend:
            narrative_changes.append(f"{n.description}: trend {prior.get('trend')} -> {n.trend}")

    new_risk_ids = {r.risk_id for r in current_risks} - prior_risk_ids
    new_risks = [r.description for r in current_risks if r.risk_id in new_risk_ids]

    assessment_change = None
    if prior_assessment is not None and prior_assessment != current_assessment:
        assessment_change = f"{prior_assessment} -> {current_assessment}"

    evidence = [f"Compared against the research pass generated at {prior_report_json.get('generated_at')}."]
    if not (sentiment_changes or narrative_changes or new_risks or assessment_change):
        evidence.append("No material change detected in narratives, risks, or overall assessment since the "
                        "prior research pass.")

    return ChangeSummary(
        has_prior_report=True, new_events=[], sentiment_change="; ".join(sentiment_changes) or None,
        assessment_change=assessment_change, new_risks=new_risks, narrative_changes=narrative_changes,
        evidence=evidence,
    )
