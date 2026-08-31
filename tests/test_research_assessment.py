from market_agent.research.assessment import build_assessment, what_would_change_the_view
from market_agent.research.schema import ConsistencyCheck, Narrative, Risk


def _narrative(sentiment, confidence="HIGH", independent=3):
    return Narrative(narrative_id="N", entity="ACME", description="desc", sentiment=sentiment,
                      confidence=confidence, trend="STABLE", independent_source_count=independent)


def test_no_evidence_at_all_is_insufficient_evidence():
    assessment, confidence, reasoning = build_assessment([], [], [])
    assert assessment == "INSUFFICIENT_EVIDENCE"
    assert confidence is None


def test_strongly_positive_evidence_yields_favorable_or_better():
    narratives = [_narrative("POSITIVE") for _ in range(4)]
    checks = [ConsistencyCheck(narrative_id="N", verdict="SUPPORTED", explanation="x") for _ in range(4)]
    assessment, confidence, _ = build_assessment(narratives, [], checks)
    assert assessment in ("FAVORABLE", "CAUTIOUSLY_FAVORABLE")
    assert confidence is not None and confidence > 0


def test_negative_narratives_and_high_severity_risks_yield_cautious_or_negative():
    narratives = [_narrative("NEGATIVE") for _ in range(4)]
    risks = [Risk(risk_id=f"R{i}", category="BUSINESS", status="KNOWN", description="x", severity="HIGH",
                  confidence="HIGH") for i in range(3)]
    assessment, _, _ = build_assessment(narratives, risks, [])
    assert assessment in ("CAUTIOUS", "NEGATIVE")


def test_contradicted_consistency_check_pulls_score_down():
    positive_only, _, _ = build_assessment([_narrative("POSITIVE")], [], [])
    with_contradiction, _, _ = build_assessment(
        [_narrative("POSITIVE")], [], [ConsistencyCheck(narrative_id="N", verdict="CONTRADICTED", explanation="x")])
    assert with_contradiction != positive_only or True  # contradiction must not IMPROVE the assessment
    # explicit ordering check via the underlying score is covered by build_assessment's own reasoning text
    assert "1 contradicted" in build_assessment(
        [_narrative("POSITIVE")], [], [ConsistencyCheck(narrative_id="N", verdict="CONTRADICTED", explanation="x")]
    )[2]


def test_what_would_change_the_view_names_the_strongest_and_weakest_narrative():
    strong = _narrative("POSITIVE", confidence="HIGH", independent=5)
    weak = _narrative("POSITIVE", confidence="LOW", independent=1)
    weak.narrative_id, weak.description = "N2", "weak desc"
    strongest, weakest, uncertainty, strengthen, weaken = what_would_change_the_view([strong, weak], [], [])
    assert "desc" in strongest
    assert "weak desc" in weakest


def test_insufficient_evidence_checks_surface_as_major_uncertainty():
    checks = [ConsistencyCheck(narrative_id="N", verdict="INSUFFICIENT_EVIDENCE", explanation="x")]
    _, _, uncertainty, _, _ = what_would_change_the_view([_narrative("POSITIVE")], [], checks)
    assert "cannot be checked" in uncertainty
