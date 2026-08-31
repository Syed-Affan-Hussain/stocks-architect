from market_agent.research.change_detection import detect_changes
from market_agent.research.schema import Narrative, Risk


def _narrative(nid, sentiment, trend="STABLE"):
    return Narrative(narrative_id=nid, entity="ACME", description=f"desc {nid}", affected_area="demand",
                      sentiment=sentiment, trend=trend, confidence="MEDIUM")


def _risk(rid):
    return Risk(risk_id=rid, category="BUSINESS", status="KNOWN", description=f"risk {rid}", severity="MEDIUM",
                confidence="MEDIUM")


def test_no_prior_report_is_disclosed_as_first_pass():
    change = detect_changes([_narrative("N1", "POSITIVE")], [], "NEUTRAL", None)
    assert change.has_prior_report is False


def test_new_narrative_detected():
    prior = {"generated_at": "t0", "assessment": "NEUTRAL", "narratives": [], "risks": []}
    change = detect_changes([_narrative("N1", "POSITIVE")], [], "NEUTRAL", prior)
    assert any("NEW narrative" in c for c in change.narrative_changes)


def test_sentiment_change_detected_via_stable_narrative_id():
    prior = {"generated_at": "t0", "assessment": "NEUTRAL",
             "narratives": [{"narrative_id": "N1", "sentiment": "POSITIVE", "trend": "STABLE"}], "risks": []}
    change = detect_changes([_narrative("N1", "NEGATIVE")], [], "NEUTRAL", prior)
    assert "POSITIVE -> NEGATIVE" in change.sentiment_change


def test_new_risk_detected():
    prior = {"generated_at": "t0", "assessment": "NEUTRAL", "narratives": [], "risks": []}
    change = detect_changes([], [_risk("R1")], "NEUTRAL", prior)
    assert change.new_risks == ["risk R1"]


def test_assessment_change_detected():
    prior = {"generated_at": "t0", "assessment": "NEUTRAL", "narratives": [], "risks": []}
    change = detect_changes([], [], "CAUTIOUS", prior)
    assert change.assessment_change == "NEUTRAL -> CAUTIOUS"


def test_no_change_reports_no_material_change():
    prior = {"generated_at": "t0", "assessment": "NEUTRAL", "narratives": [], "risks": []}
    change = detect_changes([], [], "NEUTRAL", prior)
    assert change.assessment_change is None
    assert any("No material change" in e for e in change.evidence)
