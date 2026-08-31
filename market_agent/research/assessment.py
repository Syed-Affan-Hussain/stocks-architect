"""Items 16/17: the final research assessment - a transparent, DISCLOSED
scoring rule over everything already collected (never a black box, never
"just sentiment" - item 16's explicit requirement). NOT a buy/sell signal
- see schema.py's ASSESSMENTS tuple, none of which are trading verbs.
"""
from __future__ import annotations

from market_agent.research.schema import ASSESSMENTS, ConsistencyCheck, Narrative, Risk

CONFIDENCE_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
SEVERITY_PENALTY = {"HIGH": -2, "MEDIUM": -1, "LOW": 0}


def _score(narratives: list[Narrative], risks: list[Risk], checks: list[ConsistencyCheck]) -> float:
    score = 0.0
    for n in narratives:
        sign = {"POSITIVE": 1, "NEGATIVE": -1}.get(n.sentiment, 0)
        score += sign * CONFIDENCE_WEIGHT.get(n.confidence, 1)
    for r in risks:
        score += SEVERITY_PENALTY.get(r.severity, 0)
    for c in checks:
        if c.verdict == "SUPPORTED":
            score += 1
        elif c.verdict == "CONTRADICTED":
            score -= 2
    return score


def _assessment_for_score(score: float, has_evidence: bool) -> str:
    if not has_evidence:
        return "INSUFFICIENT_EVIDENCE"
    if score >= 8:
        return "FAVORABLE"
    if score >= 3:
        return "CAUTIOUSLY_FAVORABLE"
    if score >= -2:
        return "NEUTRAL"
    if score >= -7:
        return "CAUTIOUS"
    return "NEGATIVE"


def _confidence(narratives: list[Narrative]) -> float | None:
    if not narratives:
        return None
    total_independent = sum(n.independent_source_count for n in narratives)
    breadth = min(1.0, total_independent / 10)
    conf_scores = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
    avg_conf = sum(conf_scores.get(n.confidence, 0.3) for n in narratives) / len(narratives)
    return round((breadth + avg_conf) / 2, 2)


def classify_score(score: float, has_evidence: bool) -> str:
    """Public wrapper over _assessment_for_score - same reuse rationale as
    evidence_score below: lets a blended (evidence + news-state) score be
    classified through the EXACT SAME thresholds this report already uses,
    without a second, potentially-drifting copy of those threshold values."""
    return _assessment_for_score(score, has_evidence)


def evidence_score(narratives: list[Narrative], risks: list[Risk], checks: list[ConsistencyCheck]) -> float:
    """Public wrapper over the internal composite score - added so
    research/evaluation/'s mode-B blended assessment can build on the SAME
    number build_assessment uses internally, instead of re-implementing
    this formula a second time. Does not change _score's logic or
    thresholds; this report's own assessment/build_assessment behavior is
    unaffected by this addition."""
    return _score(narratives, risks, checks)


def build_assessment(narratives: list[Narrative], risks: list[Risk], checks: list[ConsistencyCheck]
                      ) -> tuple[str, float | None, str]:
    """Returns (assessment, confidence_0_to_1, reasoning_text)."""
    has_evidence = bool(narratives) or bool(risks)
    score = _score(narratives, risks, checks)
    assessment = _assessment_for_score(score, has_evidence)
    confidence = _confidence(narratives)

    n_pos = sum(1 for n in narratives if n.sentiment == "POSITIVE")
    n_neg = sum(1 for n in narratives if n.sentiment == "NEGATIVE")
    n_high_risk = sum(1 for r in risks if r.severity == "HIGH")
    n_contradicted = sum(1 for c in checks if c.verdict == "CONTRADICTED")
    n_supported = sum(1 for c in checks if c.verdict == "SUPPORTED")

    reasoning = (
        f"Synthesized from {len(narratives)} narrative(s) ({n_pos} predominantly positive, {n_neg} predominantly "
        f"negative), {len(risks)} identified risk(s) ({n_high_risk} high-severity), and {len(checks)} "
        f"narrative-vs-fundamentals consistency check(s) ({n_supported} supported, {n_contradicted} contradicted "
        f"by disclosed data). Composite evidence score: {score:+.1f} (weighted by source confidence and "
        "independence, penalized by risk severity and contradicted narratives)."
        if has_evidence else
        "No narratives or risks were extracted from the collected evidence - insufficient basis for any "
        "directional assessment."
    )
    return assessment, confidence, reasoning


def what_would_change_the_view(narratives: list[Narrative], risks: list[Risk], checks: list[ConsistencyCheck]
                                ) -> tuple[str, str, str, str, str]:
    """Item 17. Returns (strongest_evidence, weakest_evidence,
    major_uncertainty, what_would_strengthen, what_would_weaken)."""
    ranked = sorted(narratives, key=lambda n: (CONFIDENCE_WEIGHT.get(n.confidence, 0), n.independent_source_count),
                     reverse=True)
    strongest = (f"{ranked[0].description} ({ranked[0].independent_source_count} independent source(s), "
                f"{ranked[0].confidence} confidence)." if ranked else "No narrative evidence collected.")
    weakest_candidates = [n for n in narratives if n.confidence == "LOW" or n.independent_source_count <= 1]
    weakest = (f"{weakest_candidates[0].description} (only {weakest_candidates[0].independent_source_count} "
              f"independent source(s))." if weakest_candidates else
              "No single narrative stands out as weakly evidenced relative to the others.")

    insufficient = [c for c in checks if c.verdict == "INSUFFICIENT_EVIDENCE"]
    disputed = [n for n in narratives if n.trend == "DISPUTED"]
    uncertainty_parts = []
    if insufficient:
        uncertainty_parts.append(f"{len(insufficient)} narrative(s) cannot be checked against disclosed "
                                 "fundamentals at all.")
    if disputed:
        uncertainty_parts.append(f"{len(disputed)} narrative(s) show genuinely disputed/mixed reporting.")
    major_uncertainty = " ".join(uncertainty_parts) or "No major unresolved uncertainty identified from current evidence."

    emerging_risks = [r for r in risks if r.status in ("EMERGING", "SPECULATIVE")]
    strengthen = ("Resolution of currently EMERGING/SPECULATIVE risk(s) without materializing, and additional "
                 "primary-source (SEC filing or direct company statement) confirmation of the positive "
                 "narrative(s) above." if emerging_risks else
                 "Additional primary-source confirmation of the positive narrative(s) above, and continued "
                 "fundamental performance consistent with current disclosures.")
    contradicted_supporting = [c.narrative_id for c in checks if c.verdict == "SUPPORTED"]
    weaken = ("A narrative currently SUPPORTED by fundamentals reversing in a future disclosure, or any "
             "currently EMERGING/SPECULATIVE risk materializing into a confirmed, primary-source-evidenced event."
             if contradicted_supporting or emerging_risks else
             "Deterioration in disclosed fundamentals inconsistent with the currently favorable narrative(s).")
    return strongest, weakest, major_uncertainty, strengthen, weaken
