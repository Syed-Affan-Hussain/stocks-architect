from market_agent.research.consistency import check_consistency, detect_contradictions
from market_agent.research.schema import Narrative, TimelineEvent


def _entry(start, end, val, form="10-Q", filed=None):
    return {"start": start, "end": end, "val": val, "form": form, "fp": "Q1", "fy": 2024, "filed": filed or end}


def _growing_facts():
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _entry("2023-04-02", "2023-07-01", 1000, filed="2023-08-01"),
            _entry("2024-04-02", "2024-07-01", 1500, filed="2024-08-01"),
        ]}},
    }}}


def _narrative(area, sentiment, narrative_id="N1"):
    return Narrative(narrative_id=narrative_id, entity="ACME", description="desc", affected_area=area,
                      sentiment=sentiment, trend="STABLE", confidence="MEDIUM")


def test_demand_collapsing_narrative_contradicted_by_strong_fundamentals():
    """The user's own worked example: narrative says demand is
    collapsing, fundamentals show strong growth -> CONTRADICTED."""
    narrative = _narrative("demand", "NEGATIVE")
    checks = check_consistency([narrative], _growing_facts())
    assert checks[0].verdict == "CONTRADICTED"
    assert "not supported" in checks[0].explanation.lower()


def test_positive_narrative_supported_by_growing_fundamentals():
    narrative = _narrative("revenue", "POSITIVE")
    checks = check_consistency([narrative], _growing_facts())
    assert checks[0].verdict == "SUPPORTED"


def test_non_performance_narrative_is_insufficient_evidence_not_forced():
    narrative = _narrative("management", "NEGATIVE")
    checks = check_consistency([narrative], _growing_facts())
    assert checks[0].verdict == "INSUFFICIENT_EVIDENCE"


def test_no_fundamentals_data_is_insufficient_evidence():
    narrative = _narrative("revenue", "POSITIVE")
    checks = check_consistency([narrative], None)
    assert checks[0].verdict == "INSUFFICIENT_EVIDENCE"


def test_contradiction_only_raised_when_narrative_has_both_sides():
    support_event = TimelineEvent(event_id="e1", entity="ACME", date="2024-01-01", event_type="GENERAL_NEWS",
                                   description="Demand remains strong", evidence_type="REPORTING",
                                   source_ids=["s1"], confidence="MEDIUM", materiality="MEDIUM",
                                   sentiment="POSITIVE", affected_area="demand")
    contradict_event = TimelineEvent(event_id="e2", entity="ACME", date="2024-01-02", event_type="GENERAL_NEWS",
                                      description="Customers are reducing spending", evidence_type="REPORTING",
                                      source_ids=["s2"], confidence="MEDIUM", materiality="MEDIUM",
                                      sentiment="NEGATIVE", affected_area="demand")
    narrative = Narrative(narrative_id="N1", entity="ACME", description="desc", affected_area="demand",
                           supporting_event_ids=["e1"], contradicting_event_ids=["e2"], sentiment="MIXED",
                           trend="DISPUTED", confidence="LOW")
    contradictions = detect_contradictions([narrative], {"e1": support_event, "e2": contradict_event})
    assert len(contradictions) == 1
    assert contradictions[0].side_a == "Demand remains strong"
    assert contradictions[0].side_b == "Customers are reducing spending"


def test_no_contradiction_when_narrative_has_only_one_side():
    narrative = Narrative(narrative_id="N1", entity="ACME", description="desc", affected_area="demand",
                           supporting_event_ids=["e1"], contradicting_event_ids=[], sentiment="POSITIVE",
                           trend="STABLE", confidence="LOW")
    assert detect_contradictions([narrative], {}) == []
