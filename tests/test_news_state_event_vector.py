from market_agent.research.news_state.event_vector import build_event_vector, confirmation_strength
from market_agent.research.news_state.magnitude import DIRECTION_ONLY_SCORE
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument, TimelineEvent


def _doc(source_id, reliability="TERTIARY", source_type="NEWS"):
    return SourceDocument(source_id=source_id, publisher="Pub", source_type=source_type, url="https://x",
                           published_at="2024-06-01T00:00:00+00:00", retrieved_at="2024-06-01T00:00:00+00:00",
                           entity="ACME", title="t", raw_content="c", normalized_content="c",
                           reliability=reliability, fingerprint=make_fingerprint("t", "c"))


def _event(event_id, description, sentiment, source_id="d1", materiality="MEDIUM", evidence_type="REPORTING"):
    return TimelineEvent(event_id=event_id, entity="ACME", date="2024-06-01", event_type="GENERAL_NEWS",
                          description=description, evidence_type=evidence_type, source_ids=[source_id],
                          confidence="MEDIUM", materiality=materiality, sentiment=sentiment, affected_area=None)


def test_layoffs_due_to_declining_demand_splits_into_multiple_axes():
    """The user's own worked example: one clause must simultaneously
    populate demand (negative) and risk (positive/elevated) - never
    collapse to one averaged number. No magnitude is stated, so both
    axes fall back to the disclosed DIRECTION_ONLY_SCORE, not +-1.0."""
    event = _event("e1", "Company announces layoffs due to declining demand", "NEGATIVE")
    doc = _doc("d1")
    ev = build_event_vector("ACME", [event], {"d1": doc})
    assert ev.implications["demand"] == -DIRECTION_ONLY_SCORE
    assert ev.implications["risk"] == DIRECTION_ONLY_SCORE  # elevated risk, INVERSE relationship
    assert ev.implications["profitability"] is None  # no cost-reduction language present - not fabricated
    assert ev.magnitude_confidence < 0.5  # no extracted magnitude anywhere in this event


def test_layoffs_with_explicit_cost_reduction_language_populates_profitability():
    event = _event("e1", "Company announces layoffs as part of a broader cost-cutting plan", "NEGATIVE")
    doc = _doc("d1")
    ev = build_event_vector("ACME", [event], {"d1": doc})
    assert ev.implications["profitability"] == 1.0  # the fixed, disclosed cost-reduction rule - unaffected
    assert ev.implications["risk"] == DIRECTION_ONLY_SCORE


def test_quantified_growth_scores_higher_than_unquantified_growth():
    quantified = _event("e1", "Revenue grew 40% this quarter for the company", "POSITIVE")
    unquantified = _event("e2", "Revenue increased strongly this quarter for the company", "POSITIVE")
    doc = _doc("d1")
    ev_q = build_event_vector("ACME", [quantified], {"d1": doc})
    ev_u = build_event_vector("ACME", [unquantified], {"d1": doc})
    assert ev_q.implications["growth"] == 1.0        # 40% saturates the percent anchor table
    assert ev_u.implications["growth"] == DIRECTION_ONLY_SCORE
    assert ev_q.implications["growth"] > ev_u.implications["growth"]
    assert ev_q.magnitude_confidence > ev_u.magnitude_confidence


def test_small_and_large_quantified_moves_score_differently():
    """The core fix: a 2% move and a 40% move must not both collapse to
    the same value."""
    small = _event("e1", "Revenue grew 2% this quarter for the company", "POSITIVE")
    large = _event("e2", "Revenue grew 40% this quarter for the company", "POSITIVE")
    doc = _doc("d1")
    ev_small = build_event_vector("ACME", [small], {"d1": doc})
    ev_large = build_event_vector("ACME", [large], {"d1": doc})
    assert ev_small.implications["growth"] != ev_large.implications["growth"]
    assert ev_small.implications["growth"] < ev_large.implications["growth"]


def test_negative_percent_growth_still_signed_correctly():
    event = _event("e1", "Revenue declined 20% this quarter for the company", "NEGATIVE")
    doc = _doc("d1")
    ev = build_event_vector("ACME", [event], {"d1": doc})
    assert ev.implications["growth"] == -0.73  # sign from the lexicon ("declined"), size from the 20% anchor


def test_sec_filing_event_is_observed_fact_with_high_certainty():
    event = _event("e1", "Revenue increased to $30 billion for the quarter", "POSITIVE")
    doc = _doc("d1", reliability="PRIMARY", source_type="SEC_FILING")
    ev = build_event_vector("ACME", [event], {"d1": doc})
    assert ev.epistemic_status == "OBSERVED_FACT"
    assert ev.certainty == 1.0


def test_management_guidance_claim_has_lower_certainty_than_observed_fact():
    filing_event = _event("e1", "Revenue increased to $30 billion for the quarter", "POSITIVE")
    filing_doc = _doc("d1", reliability="PRIMARY", source_type="SEC_FILING")
    claim_event = _event("e2", "Management raised its guidance for the year", "POSITIVE", source_id="d2")
    claim_doc = _doc("d2")
    fact_vector = build_event_vector("ACME", [filing_event], {"d1": filing_doc})
    claim_vector = build_event_vector("ACME", [claim_event], {"d2": claim_doc})
    assert fact_vector.certainty > claim_vector.certainty


def test_independent_source_count_excludes_syndicated_duplicates():
    canonical = _doc("d1")
    duplicate = SourceDocument(source_id="d2", publisher="Other", source_type="NEWS", url="https://y",
                                published_at="2024-06-01T00:00:00+00:00", retrieved_at="2024-06-01T00:00:00+00:00",
                                entity="ACME", title="t2", raw_content="c2", normalized_content="c2",
                                reliability="TERTIARY", fingerprint=make_fingerprint("t2", "c2"), duplicate_of="d1")
    event1 = _event("e1", "Revenue grew 10% this quarter for the company", "POSITIVE", source_id="d1")
    event2 = _event("e2", "Revenue grew 10% this quarter for the company", "POSITIVE", source_id="d2")
    ev = build_event_vector("ACME", [event1, event2], {"d1": canonical, "d2": duplicate})
    assert ev.independent_source_count == 1


def test_confirmation_strength_saturates_not_scales_linearly():
    """Diminishing returns - going from 1 to 2 sources should add more
    confirmation strength than going from 4 to 5."""
    s1 = confirmation_strength(1, 1.0)
    s2 = confirmation_strength(2, 1.0)
    s4 = confirmation_strength(4, 1.0)
    s5 = confirmation_strength(5, 1.0)
    assert 0 < s1 < s2 < s4 < s5 <= 1.0
    assert (s2 - s1) > (s5 - s4)  # diminishing returns


def test_confirmation_strength_zero_sources_is_zero():
    assert confirmation_strength(0, 1.0) == 0.0


def test_speculative_clause_counts_for_less_than_a_reported_one():
    """MODALITY-WEIGHTED mean (SENTiVENT's modality attribute - see
    event_vector.py's MODALITY_WEIGHT docstring): a REPORTING clause
    saying revenue grew and a SPECULATION clause hedging that revenue
    could later decline should NOT average to a flat 0.0 - the stated
    clause must dominate the hedged one, not cancel it out 1:1."""
    reported = _event("e1", "Revenue grew this quarter for the company", "POSITIVE",
                       evidence_type="REPORTING")
    speculative = _event("e2", "Revenue could decline next quarter for the company", "NEGATIVE",
                          source_id="d1", evidence_type="SPECULATION")
    doc = _doc("d1")
    ev = build_event_vector("ACME", [reported, speculative], {"d1": doc})
    assert ev.implications["growth"] == 0.21  # not 0.0 - see the precomputed weighted mean above
    assert ev.implications["growth"] > 0


def test_risk_axis_is_always_inferred_never_stated():
    """risk only ever fires via the INVERSE relationship in
    IMPLICATION_RULES - no news clause directly states a "risk" figure,
    it is always derived from some other reported fact."""
    event = _event("e1", "Company announces layoffs due to declining demand", "NEGATIVE")
    doc = _doc("d1")
    ev = build_event_vector("ACME", [event], {"d1": doc})
    assert ev.implication_basis["risk"] == "INFERRED"
    assert ev.implication_basis["demand"] == "STATED"


def test_profitability_basis_is_mixed_when_stated_and_inferred_both_contribute():
    """One clause directly states earnings news (STATED, via the DIRECT
    "earnings"->profitability rule); a separate clause in the same
    cluster only IMPLIES a profitability effect via the cost-reduction
    carve-out. Both landing on the same axis must show as MIXED, not
    silently collapse to either label alone."""
    earnings_clause = _event("e1", "Earnings rose this quarter for the company", "POSITIVE")
    cost_cutting_clause = _event("e2", "Company announces layoffs as part of a broader cost-cutting plan",
                                  "NEGATIVE", source_id="d2")
    docs = {"d1": _doc("d1"), "d2": _doc("d2")}
    ev = build_event_vector("ACME", [earnings_clause, cost_cutting_clause], docs)
    assert ev.implication_basis["profitability"] == "MIXED"
