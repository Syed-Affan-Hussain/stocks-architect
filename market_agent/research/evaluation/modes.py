"""Computes the three comparable prediction modes from ONE already-run
ResearchReport - never a second, independent data-collection pass (that
would triple the network calls per entity and risk the three modes seeing
slightly different underlying news/filings if run at different moments).
All three modes are computed from the SAME collected evidence; they differ
only in which of it each mode is ALLOWED to use.

  A_NO_NEWS  - the existing model exactly as it already runs. Uses
               ResearchReport.assessment/assessment_confidence UNCHANGED -
               assessment.py has never consumed news_state (see its own
               module docstring), so this mode requires no new code at
               all, only a decision_mapping.py pass over its existing
               output.
  B_BLENDED  - the existing model's narrative/risk/consistency evidence
               score PLUS a news-state-derived term, reclassified through
               the SAME thresholds assessment.py already uses (see
               assessment.py's classify_score/evidence_score - added as
               public wrappers, not a second copy of that logic).
  C_NEWS_ONLY - CompanyNewsState's own quantified axes, with NO
               narrative/risk/consistency evidence at all - isolates
               whether the news quantifier carries signal on its own.

NEWS_STATE_BLEND_WEIGHT below is THIS module's own disclosed calibration,
not literature-specified - see its own comment.
"""
from __future__ import annotations

from dataclasses import dataclass

from market_agent.research import assessment as assessment_module
from market_agent.research.evaluation.decision_mapping import (
    MappedDecision, map_assessment_to_decision, news_state_to_decision,
)
from market_agent.research.schema import ResearchReport

MODE_A = "A_NO_NEWS"
MODE_B = "B_BLENDED"
MODE_C = "C_NEWS_ONLY"
MODES = (MODE_A, MODE_B, MODE_C)

# A blended score adds news_term = mean(non-null CompanyNewsState axis values) * this weight to
# assessment.py's existing evidence_score() before reclassifying. Chosen so ONE fully-saturated
# (+-1.0) news axis contributes roughly as much as ONE HIGH-confidence narrative already does to
# evidence_score() (CONFIDENCE_WEIGHT["HIGH"]=3 there) - a deliberate, disclosed order-of-magnitude
# choice so news can meaningfully move the blended assessment without silently dominating it, NOT a
# value fit to any outcome data (no such data exists yet - see this package's outcome_resolution.py).
NEWS_STATE_BLEND_WEIGHT = 3.0


@dataclass(frozen=True)
class ModeResult:
    mode: str
    decision: MappedDecision
    reasoning: str


def _news_state_term(news_state: dict | None) -> tuple[float, bool]:
    if news_state is None:
        return 0.0, False
    dims = [v for v in news_state.get("dimensions", {}).values() if v is not None]
    if not dims:
        return 0.0, False
    return (sum(dims) / len(dims)) * NEWS_STATE_BLEND_WEIGHT, True


def compute_mode_a(report: ResearchReport) -> ModeResult:
    decision = map_assessment_to_decision(report.assessment, report.assessment_confidence)
    return ModeResult(mode=MODE_A, decision=decision,
                       reasoning=f"Unmodified existing assessment: {report.assessment_reasoning}")


def compute_mode_b(report: ResearchReport) -> ModeResult:
    base_score = assessment_module.evidence_score(report.narratives, report.risks, report.consistency_checks)
    news_term, news_had_signal = _news_state_term(report.news_state)
    blended_score = base_score + news_term
    has_evidence = bool(report.narratives) or bool(report.risks) or news_had_signal
    assessment = assessment_module.classify_score(blended_score, has_evidence)
    decision = map_assessment_to_decision(assessment, report.assessment_confidence)
    reasoning = (f"Blended score = evidence_score({base_score:+.1f}) + news_term({news_term:+.1f}, "
                f"weight={NEWS_STATE_BLEND_WEIGHT}) = {blended_score:+.1f} -> {assessment}.")
    return ModeResult(mode=MODE_B, decision=decision, reasoning=reasoning)


def compute_mode_c(report: ResearchReport) -> ModeResult:
    decision = news_state_to_decision(report.news_state)
    return ModeResult(mode=MODE_C, decision=decision,
                       reasoning="Derived purely from CompanyNewsState.dimensions - no narrative/risk/"
                                 "consistency evidence used.")


def compute_all_modes(report: ResearchReport) -> list[ModeResult]:
    return [compute_mode_a(report), compute_mode_b(report), compute_mode_c(report)]
