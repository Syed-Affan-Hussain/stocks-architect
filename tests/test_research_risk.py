from market_agent.research.risk import extract_all_risks, extract_risks_from_fundamentals
from market_agent.research.schema import Contradiction, FundamentalFact, Narrative, TimelineEvent


def _event(event_id, area, sentiment="NEGATIVE", evidence_type="REPORTING"):
    return TimelineEvent(event_id=event_id, entity="ACME", date="2024-01-01", event_type="GENERAL_NEWS",
                          description="desc", evidence_type=evidence_type, source_ids=["s1"], confidence="MEDIUM",
                          materiality="MEDIUM", sentiment=sentiment, affected_area=area)


def test_negative_narrative_becomes_a_risk_with_mapped_category():
    n = Narrative(narrative_id="N1", entity="ACME", description="Regulatory scrutiny increasing",
                  affected_area="regulatory", sentiment="NEGATIVE", trend="EMERGING", confidence="MEDIUM",
                  supporting_event_ids=["e1"], independent_source_count=2)
    events_by_id = {"e1": _event("e1", "regulatory")}
    risks = extract_all_risks([n], events_by_id, [], [], None)
    assert len(risks) == 1
    assert risks[0].category == "REGULATORY"


def test_positive_narrative_does_not_become_a_risk():
    n = Narrative(narrative_id="N1", entity="ACME", description="Strong demand", affected_area="demand",
                  sentiment="POSITIVE", trend="STABLE", confidence="MEDIUM")
    assert extract_all_risks([n], {}, [], [], None) == []


def test_contradiction_becomes_a_narrative_category_risk():
    c = Contradiction(contradiction_id="C1", description="conflicting reports", side_a="a", side_a_source_ids=["s1"],
                       side_b="b", side_b_source_ids=["s2"], what_would_resolve_it="more evidence")
    risks = extract_all_risks([], {}, [c], [], None)
    assert len(risks) == 1
    assert risks[0].category == "NARRATIVE"


def test_high_debt_to_cash_ratio_flagged_as_balance_sheet_risk():
    facts = [
        FundamentalFact(tag="LongTermDebt", label="Long-term debt", period_end="2024-01-01", value=9_000_000_000,
                         unit="USD", fiscal_period_type="Q"),
        FundamentalFact(tag="CashAndCashEquivalents", label="Cash", period_end="2024-01-01", value=1_000_000_000,
                         unit="USD", fiscal_period_type="Q"),
    ]
    risks = extract_risks_from_fundamentals(facts, "src1")
    assert any(r.category == "BALANCE_SHEET" for r in risks)


def test_low_debt_to_cash_ratio_not_flagged():
    facts = [
        FundamentalFact(tag="LongTermDebt", label="Long-term debt", period_end="2024-01-01", value=1_000_000_000,
                         unit="USD", fiscal_period_type="Q"),
        FundamentalFact(tag="CashAndCashEquivalents", label="Cash", period_end="2024-01-01", value=20_000_000_000,
                         unit="USD", fiscal_period_type="Q"),
    ]
    assert extract_risks_from_fundamentals(facts, "src1") == []


def test_negative_free_cash_flow_flagged_as_liquidity_risk():
    facts = [FundamentalFact(tag="FreeCashFlow", label="FCF", period_end="2024-01-01", value=-500_000_000,
                              unit="USD", fiscal_period_type="Q")]
    risks = extract_risks_from_fundamentals(facts, "src1")
    assert len(risks) == 1
    assert risks[0].category == "LIQUIDITY"
