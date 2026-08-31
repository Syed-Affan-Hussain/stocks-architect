"""Knowledge-state dashboard CLI - stage 5 item 8's "report command" over
reporting/knowledge_state.py's already-tested `build_knowledge_state_report`.
This module only formats and prints; it computes nothing new itself.

    python -m market_agent.report --ledger data_cache/prediction_ledger.sqlite
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from market_agent.reporting.knowledge_state import KnowledgeStateReport, build_knowledge_state_report
from market_agent.store import db


def format_report(report: KnowledgeStateReport) -> str:
    lines = [f"KNOWLEDGE STATE - generated {report.generated_at}",
              f"knowledge_version: {report.knowledge_version}", f"LLM status: {report.llm_status}", ""]

    if report.operational_counts is not None:
        c = report.operational_counts
        lines += ["OPERATIONAL COUNTS",
                   f"  events ingested: {c.n_events_ingested}  |  resolved: {c.n_events_resolved}  |  "
                   f"insufficient-precedent: {c.n_events_insufficient_precedent}  |  data-error: {c.n_events_data_error}",
                   f"  error type distribution (resolved only): {c.error_type_distribution or '{}'}",
                   f"  hypotheses: {c.n_hypotheses_generated} generated, {c.n_hypotheses_confirmed} confirmed, "
                   f"{c.n_hypotheses_rejected} rejected, {c.n_hypotheses_untested} untested",
                   f"  relationships: {c.n_relationships_active} active, {c.n_relationships_shadow} shadow, "
                   f"{c.n_relationships_retired} retired  |  governance changes: {c.n_governance_changes}", ""]

    lines.append(f"ACTIVE RELATIONSHIPS ({len(report.active_relationships)})")
    for r in report.active_relationships:
        lines.append(f"  {r.relationship_id}  {r.condition}  effect={r.effect_estimate:+.2%}  "
                       f"N={r.n_supporting}  decay={r.decay_state}  "
                       f"supported={r.n_predictions_supported}/contradicted={r.n_predictions_contradicted}")

    lines.append(f"SHADOW RELATIONSHIPS ({len(report.shadow_relationships)})")
    for r in report.shadow_relationships:
        lines.append(f"  {r.relationship_id}  {r.condition}  effect={r.effect_estimate:+.2%}  N={r.n_supporting}")

    lines.append(f"RETIRED RELATIONSHIPS ({len(report.retired_relationships)})")
    for r in report.retired_relationships:
        lines.append(f"  {r.relationship_id}  {r.condition}")

    lines.append("")
    lines.append("CALIBRATION BY HORIZON (STATIC vs ADAPTIVE)")
    for hc in report.calibration_by_horizon:
        static_acc = f"{hc.static_direction_accuracy:.1%}" if hc.static_direction_accuracy is not None else "n/a"
        adaptive_acc = f"{hc.adaptive_direction_accuracy:.1%}" if hc.adaptive_direction_accuracy is not None else "n/a"
        static_mae = f"{hc.static_mae:.4f}" if hc.static_mae is not None else "n/a"
        adaptive_mae = f"{hc.adaptive_mae:.4f}" if hc.adaptive_mae is not None else "n/a"
        lines.append(f"  {hc.horizon_days}D: STATIC n={hc.static_n} dir_acc={static_acc} mae={static_mae}  |  "
                       f"ADAPTIVE n={hc.adaptive_n} dir_acc={adaptive_acc} mae={adaptive_mae}")

    lines.append("")
    lines.append("SOURCE RELIABILITY")
    for s in report.source_reliability:
        hit_rate = f"{s.hit_rate:.1%}" if s.hit_rate is not None else "n/a"
        lines.append(f"  {s.source}: n={s.n_resolved_predictions}  learnable_errors={s.n_learnable_errors}  "
                       f"hit_rate={hit_rate}")

    lines.append("")
    lines.append(f"REJECTED HYPOTHESES ({len(report.rejected_hypotheses)})")
    for h in report.rejected_hypotheses[:20]:
        lines.append(f"  {h['hypothesis_id']}  {h['condition']}  reason={h['reason']}  n={h['n']}")
    if len(report.rejected_hypotheses) > 20:
        lines.append(f"  ... and {len(report.rejected_hypotheses) - 20} more")

    lines.append("")
    lines.append("TRADING CONCEPTS (UNTESTED / REJECTED / SHADOW / ACTIVE / DECAYING / RETIRED)")
    for c in report.concepts:
        n_untested = len(c.untested_hypotheses)
        n_rejected = len(c.rejected_hypotheses)
        n_shadow = len(c.shadow_relationships)
        n_active = len(c.active_relationships)
        n_decaying = len(c.decaying_relationships)
        n_retired = len(c.retired_relationships)
        if n_untested + n_rejected + n_shadow + n_active + n_decaying + n_retired == 0:
            continue  # nothing has ever touched this concept - skip the empty line, not hide the fact
        flag = "" if c.computable else "  [NOT COMPUTABLE - see concepts/ontology.py]"
        lines.append(f"  {c.concept}{flag}: untested={n_untested} rejected={n_rejected} shadow={n_shadow} "
                     f"active={n_active} decaying={n_decaying} retired={n_retired}")
        for r in c.active_relationships + c.decaying_relationships:
            state = "DECAYING" if r.is_decaying else "ACTIVE"
            ci = f"[{r.ci_low}, {r.ci_high}]" if r.ci_low is not None else "n/a"
            lines.append(f"      [{state}] {r.relationship_id}  {r.condition}  effect={r.effect_estimate:+.2%}  "
                         f"N={r.n_supporting}  CI={ci}  out_of_sample={r.n_predictions_supported} "
                         f"(contradicted={r.n_predictions_contradicted})  methodologies={r.methodology_ids}")
        if c.contributing_methodologies:
            names = ", ".join(f"{m['name']} ({m['practitioner']})" for m in c.contributing_methodologies)
            lines.append(f"      contributing methodologies: {names}")

    lines.append("")
    lines.append(f"METHODOLOGIES ({len(report.methodologies)})")
    for m in report.methodologies:
        lines.append(f"  {m.name} ({m.practitioner}, {m.source_type}, via {m.extractor_name})")
        lines.append(f"      concepts claimed: {m.concepts_claimed}")
        lines.append(f"      concepts with active evidence: {m.concepts_with_active_evidence}")
        lines.append(f"      concepts with no active evidence yet: {m.concepts_with_no_active_evidence}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m market_agent.report",
                                      description="Print the current knowledge-state dashboard.")
    parser.add_argument("--ledger", default="data_cache/prediction_ledger.sqlite",
                         help="sqlite path for the persistent ledger")
    args = parser.parse_args(argv)

    conn = db.connect(args.ledger)
    report = build_knowledge_state_report(conn, now=datetime.now(timezone.utc))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
