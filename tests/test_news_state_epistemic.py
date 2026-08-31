from market_agent.research.news_state.epistemic import (
    certainty_for_breakdown, classify_epistemic_status, dominant_status,
)


def test_sec_filing_is_always_observed_fact():
    assert classify_epistemic_status("Revenue increased.", "SEC_FILING") == "OBSERVED_FACT"


def test_management_attribution_is_management_claim():
    assert classify_epistemic_status("Management raised its guidance for the year.", "NEWS") == "MANAGEMENT_CLAIM"


def test_speculation_wins_over_management_attribution():
    """'Management expects a possible decline' is still speculative
    content even though management is the speaker - hedged forward claims
    stay SPECULATION regardless of who is talking."""
    assert classify_epistemic_status("Management expects results could decline next quarter.", "NEWS") == "SPECULATION"


def test_analyst_interpretation_detected():
    assert classify_epistemic_status("The drop in orders suggests weakening demand.", "NEWS") == "ANALYST_INTERPRETATION"


def test_default_is_third_party_reporting():
    assert classify_epistemic_status("The company held its annual meeting today.", "NEWS") == "THIRD_PARTY_REPORTING"


def test_certainty_weight_ordering():
    fact_only = certainty_for_breakdown({"OBSERVED_FACT": 1})
    claim_only = certainty_for_breakdown({"MANAGEMENT_CLAIM": 1})
    speculation_only = certainty_for_breakdown({"SPECULATION": 1})
    assert fact_only > claim_only > speculation_only


def test_certainty_empty_breakdown_is_zero():
    assert certainty_for_breakdown({}) == 0.0


def test_dominant_status_prefers_strongest_not_most_frequent():
    breakdown = {"SPECULATION": 9, "OBSERVED_FACT": 1}
    assert dominant_status(breakdown) == "OBSERVED_FACT"


def test_dominant_status_empty_defaults_to_third_party_reporting():
    assert dominant_status({}) == "THIRD_PARTY_REPORTING"
