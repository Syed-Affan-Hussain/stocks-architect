"""Item 21: renders a ResearchReport as the CLI text format specified by
the product spec. The SAME report also exposes .to_dict() (schema.py) for
a future frontend to consume - this module only handles the text view.
"""
from __future__ import annotations

from market_agent.research.schema import ResearchReport

RULE = "━" * 40


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _render_news_state_lines(ns: dict | None) -> list[str]:
    """Isolated from render_text so it can be unit-tested against a hand-built
    news_state dict (matching news_state.schema.CompanyNewsState.to_dict()'s
    real shape) without constructing a full ResearchReport."""
    if ns is None:
        return ["  SOURCE_UNAVAILABLE - no news retrieved this pass."]
    lines: list[str] = []
    dims = {k: v for k, v in ns["dimensions"].items() if v is not None}
    if dims:
        lines.append(f"  Signed axes [-1,1], from {ns['independent_event_count']} independent event(s) "
                     f"(disclosed heuristic, not a calibrated probability):")
        for axis, value in sorted(dims.items()):
            lines.append(f"    {axis}: {value:+.2f}")
    else:
        lines.append("  No axis carried a signal this pass (no extractable event matched a tracked area).")
    if ns["contradiction_axes"]:
        lines.append(f"  Contradiction: {', '.join(ns['contradiction_axes'])} - sources disagree; "
                     "the value above is a midpoint, not a resolved consensus.")
    lines.append(f"  Confidence: {_pct(ns['confidence'])}  |  News volume: {ns['news_volume']} document(s)  |  "
                 f"Half-life: {ns['half_life_days']:.0f}d")
    if ns.get("state_change"):
        change_str = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(ns["state_change"].items()))
        lines.append(f"  Change vs. prior pass: {change_str}")
    if ns.get("excluded_by_role"):
        excl_str = ", ".join(f"{role.lower()}: {count}" for role, count in sorted(ns["excluded_by_role"].items()))
        lines.append(f"  Excluded (not attributed to this company - competitor/industry news seen but "
                     f"not counted above): {excl_str}")
    return lines


def render_text(report: ResearchReport) -> str:
    lines: list[str] = []
    lines.append(RULE)
    lines.append(f"{report.profile.name or report.entity} — RESEARCH REPORT")
    lines.append(f"Assessment:\n  {report.assessment.replace('_', ' ')}")
    lines.append(f"Confidence:\n  {_pct(report.assessment_confidence)}")
    lines.append(f"Research period:\n  Last {report.research_period_days} days")
    lines.append(f"LLM status:\n  {report.llm_status}")
    lines.append(RULE)

    lines.append("EXECUTIVE SUMMARY")
    lines.append(report.executive_summary)
    lines.append("")

    lines.append("WHAT CHANGED")
    if report.change.has_prior_report:
        for e in report.change.evidence:
            lines.append(f"  {e}")
        if report.change.assessment_change:
            lines.append(f"  Assessment: {report.change.assessment_change}")
        if report.change.sentiment_change:
            lines.append(f"  Sentiment: {report.change.sentiment_change}")
        for c in report.change.new_risks:
            lines.append(f"  New risk: {c}")
        for c in report.change.narrative_changes:
            lines.append(f"  {c}")
    else:
        lines.append("  First research pass for this company - no prior report to compare against.")
    lines.append("")

    lines.append("KEY DEVELOPMENTS")
    material = [e for e in report.timeline if e.materiality == "HIGH"][:8]
    if material:
        for e in material:
            lines.append(f"  [{e.date}] ({e.evidence_type}) {e.description}")
    else:
        lines.append("  No high-materiality developments identified in this pass.")
    lines.append("")

    lines.append("NEWS SENTIMENT")
    for label, sentiment in (("Positive", "POSITIVE"), ("Negative", "NEGATIVE"), ("Mixed", "MIXED")):
        items = [n.description for n in report.narratives if n.sentiment == sentiment]
        lines.append(f"  {label}:")
        if items:
            for i in items:
                lines.append(f"    - {i}")
        else:
            lines.append("    (none)")
    lines.append("")

    lines.append("MAJOR NARRATIVES")
    if report.narratives:
        for i, n in enumerate(report.narratives[:10], 1):
            lines.append(f"  {i}. {n.description}")
            lines.append(f"     trend={n.trend}  sentiment={n.sentiment}  confidence={n.confidence}  "
                        f"independent_sources={n.independent_source_count} (of {n.source_count} total)")
    else:
        lines.append("  No narratives identified.")
    lines.append("")

    lines.append("FUNDAMENTALS")
    if report.profile.fundamentals:
        for f in report.profile.fundamentals:
            if not isinstance(f.value, (int, float)):
                value_str = "SOURCE_UNAVAILABLE"
            elif abs(f.value) < 1000:  # a per-share figure (EPS) - integer formatting would silently lose precision
                value_str = f"{f.value:,.2f}"
            else:
                value_str = f"{f.value:,.0f}"
            lines.append(f"  {f.label}: {value_str}" + (f" (as of {f.period_end})" if f.period_end else ""))
    else:
        lines.append("  SOURCE_UNAVAILABLE - no disclosed fundamentals retrieved.")
    lines.append("")

    lines.append("MARKET CONTEXT")
    if report.market_context:
        lines.append(f"  {report.market_context.narrative_text}")
    else:
        lines.append("  SOURCE_UNAVAILABLE - no price history retrieved.")
    lines.append("")

    lines.append("QUANTIFIED NEWS STATE")
    lines.extend(_render_news_state_lines(report.news_state))
    lines.append("")

    lines.append("NEWS VS FUNDAMENTALS")
    if report.consistency_checks:
        for c in report.consistency_checks:
            lines.append(f"  [{c.verdict}] {c.explanation}")
    else:
        lines.append("  No consistency checks performed (no narratives to check).")
    lines.append("")

    lines.append("RISKS")
    if report.risks:
        for r in report.risks:
            lines.append(f"  [{r.severity}/{r.status}/{r.category}] {r.description}")
    else:
        lines.append("  No structured risks identified in this pass.")
    lines.append("")

    lines.append("CONTRADICTIONS")
    if report.contradictions:
        for c in report.contradictions:
            lines.append(f"  {c.description}")
            lines.append(f"    Side A: {c.side_a}")
            lines.append(f"    Side B: {c.side_b}")
            lines.append(f"    Would resolve: {c.what_would_resolve_it}")
    else:
        lines.append("  No explicit contradictions detected in this pass.")
    lines.append("")

    lines.append("HISTORICAL CONTEXT")
    if report.historical_reactions:
        for h in report.historical_reactions:
            lines.append(f"  {h.event_type}/{h.direction} at {h.horizon_days}D: median {h.median_reaction:+.2%}, "
                        f"positive in {_pct(h.pct_positive)} of N={h.n} cases (descriptive, cross-company - "
                        "not a prediction for this company).")
    else:
        lines.append("  No comparable historical event data available.")
    lines.append("")

    lines.append("WHAT WOULD CHANGE THE VIEW")
    lines.append(f"  Strongest evidence: {report.strongest_evidence}")
    lines.append(f"  Weakest evidence: {report.weakest_evidence}")
    lines.append(f"  Major uncertainty: {report.major_uncertainty}")
    lines.append(f"  Would strengthen: {report.what_would_strengthen}")
    lines.append(f"  Would weaken: {report.what_would_weaken}")
    lines.append("")

    lines.append("WHAT TO WATCH NEXT")
    for w in report.what_to_watch:
        lines.append(f"  - {w}")
    lines.append("")

    lines.append("SOURCES")
    lines.append(f"  {len(report.sources)} document(s) collected "
                 f"({len({s.duplicate_of or s.source_id for s in report.sources})} independent).")
    if report.unavailable_sources:
        lines.append(f"  UNAVAILABLE: {', '.join(report.unavailable_sources)}")
    for s in report.sources[:15]:
        marker = " (duplicate)" if s.duplicate_of else ""
        lines.append(f"  [{s.reliability}] {s.publisher}: {s.title}{marker}")
        if s.url:
            lines.append(f"      {s.url}")

    lines.append(RULE)
    return "\n".join(lines)
