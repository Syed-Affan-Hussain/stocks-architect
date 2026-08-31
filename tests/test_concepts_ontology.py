from market_agent.concepts.ontology import CONCEPT_REGISTRY, COMPUTABLE_CONCEPTS, TradingConcept

EXPECTED_CONCEPTS = {
    "TREND", "MOMENTUM", "MEAN_REVERSION", "BREAKOUT", "PULLBACK", "PRICE_ACTION",
    "SUPPORT_RESISTANCE", "VOLATILITY_COMPRESSION_EXPANSION", "VOLUME", "RELATIVE_VOLUME", "VWAP",
    "MOVING_AVERAGE_STRUCTURE", "MARKET_STRUCTURE", "GAPS", "OPENING_RANGE", "RELATIVE_STRENGTH",
    "SECTOR_CONTEXT", "MULTI_TIMEFRAME_CONFIRMATION", "CATALYST_EVENT_REACTION", "RISK_MANAGEMENT",
    # stage 7 item 7 - see ontology.py's module docstring for why these two, and why ma_slope_state
    # deliberately does NOT get its own concept here.
    "CLOSE_LOCATION_VALUE", "LIQUIDITY_REGIME",
}


def test_ontology_covers_exactly_the_twenty_two_requested_categories():
    assert {c.value for c in TradingConcept} == EXPECTED_CONCEPTS
    assert len(TradingConcept) == 22


def test_every_concept_has_a_registry_entry():
    for concept in TradingConcept:
        assert concept in CONCEPT_REGISTRY
        entry = CONCEPT_REGISTRY[concept]
        assert entry.concept == concept
        assert entry.description
        assert entry.computation_note


def test_non_computable_concepts_are_disclosed_not_silently_dropped():
    non_computable = {c for c, d in CONCEPT_REGISTRY.items() if not d.computable}
    assert non_computable == {TradingConcept.OPENING_RANGE, TradingConcept.SECTOR_CONTEXT,
                               TradingConcept.RISK_MANAGEMENT}
    for concept in non_computable:
        note = CONCEPT_REGISTRY[concept].computation_note
        assert "NOT COMPUTABLE" in note or "NOT a market-state signal" in note


def test_computable_concepts_tuple_matches_registry():
    assert set(COMPUTABLE_CONCEPTS) == {c for c, d in CONCEPT_REGISTRY.items() if d.computable}
    assert len(COMPUTABLE_CONCEPTS) == 19
