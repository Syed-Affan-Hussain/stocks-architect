from market_agent.research.evaluation.modes import MODE_A, MODE_B, MODE_C, compute_all_modes
from market_agent.research.schema import ChangeSummary, CompanyProfile, ConsistencyCheck, Narrative, ResearchReport, Risk


def _report(assessment="NEUTRAL", assessment_confidence=0.5, narratives=(), risks=(), consistency_checks=(),
            news_state=None):
    return ResearchReport(
        entity="ACME", generated_at="2024-06-01T00:00:00+00:00", llm_status="UNAVAILABLE",
        research_period_days=30, assessment=assessment, assessment_confidence=assessment_confidence,
        assessment_reasoning="test reasoning", executive_summary="test summary",
        profile=CompanyProfile(entity="ACME", cik=None, name="Acme Corp"), timeline=[],
        narratives=list(narratives), consistency_checks=list(consistency_checks), contradictions=[],
        risks=list(risks), positive_factors=[], market_context=None, historical_reactions=[],
        change=ChangeSummary(has_prior_report=False), strongest_evidence="none", weakest_evidence="none",
        major_uncertainty="none", what_would_strengthen="none", what_would_weaken="none", what_to_watch=[],
        sources=[], news_state=news_state,
    )


def _narrative(sentiment="POSITIVE", confidence="HIGH"):
    return Narrative(narrative_id="n1", entity="ACME", description="d", affected_area="earnings",
                      sentiment=sentiment, trend="EMERGING", confidence=confidence, source_count=3,
                      independent_source_count=3)


def _news_state(dimensions, confidence=0.4, contradiction_axes=()):
    return {"dimensions": dimensions, "confidence": confidence, "contradiction_axes": list(contradiction_axes)}


def test_mode_a_uses_the_reports_own_assessment_unchanged():
    report = _report(assessment="FAVORABLE", assessment_confidence=0.7,
                      news_state=_news_state({"growth": -1.0}))  # deliberately opposite-signed news
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_A].decision.predicted_impact == 1.0  # unaffected by the contradicting news_state
    assert results[MODE_A].decision.decision_label == "FAVORABLE"


def test_mode_c_ignores_the_assessment_entirely():
    report = _report(assessment="NEGATIVE", assessment_confidence=0.9,
                      news_state=_news_state({"growth": 1.0, "demand": 1.0}))
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_C].decision.predicted_impact == 1.0  # purely from news_state, opposite of the assessment
    assert results[MODE_C].decision.decision_label == "NEWS_ONLY_UP"


def test_mode_b_blends_evidence_and_news_in_the_same_direction():
    """Strong positive narrative evidence plus strong positive news should
    push mode B at least as favorable as mode A alone."""
    report = _report(assessment="CAUTIOUSLY_FAVORABLE", assessment_confidence=0.6,
                      narratives=[_narrative("POSITIVE", "HIGH")],
                      news_state=_news_state({"growth": 1.0}))
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_B].decision.predicted_impact >= results[MODE_A].decision.predicted_impact


def test_mode_b_can_diverge_from_mode_a_when_news_disagrees():
    """Weak positive evidence (CAUTIOUSLY_FAVORABLE, score near the
    NEUTRAL boundary) combined with strongly negative news should be able
    to pull the blended classification down a notch from mode A's."""
    report = _report(assessment="CAUTIOUSLY_FAVORABLE", assessment_confidence=0.5,
                      narratives=[_narrative("POSITIVE", "MEDIUM")],  # score=2, just below CAUTIOUSLY_FAVORABLE's own 3 threshold in isolation
                      news_state=_news_state({"growth": -1.0, "demand": -1.0}))
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_B].decision.predicted_impact < results[MODE_A].decision.predicted_impact


def test_mode_b_with_no_news_state_matches_mode_a_style_score():
    """No news at all (news_state=None) - mode B's blended score should
    reduce to exactly the existing evidence_score with a zero news term,
    landing on the SAME classification as mode A when driven by the same
    narratives/risks/checks."""
    report = _report(assessment="FAVORABLE", assessment_confidence=0.8,
                      narratives=[_narrative("POSITIVE", "HIGH")] * 3,  # score=9 -> FAVORABLE on its own
                      news_state=None)
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_B].decision.decision_label == results[MODE_A].decision.decision_label == "FAVORABLE"


def test_all_three_modes_are_always_present():
    report = _report()
    modes_seen = {r.mode for r in compute_all_modes(report)}
    assert modes_seen == {MODE_A, MODE_B, MODE_C}


def test_insufficient_evidence_assessment_yields_no_signal_in_mode_a_and_b():
    report = _report(assessment="INSUFFICIENT_EVIDENCE", assessment_confidence=None, news_state=None)
    results = {r.mode: r for r in compute_all_modes(report)}
    assert results[MODE_A].decision.predicted_impact is None
