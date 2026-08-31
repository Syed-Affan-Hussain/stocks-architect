import pytest

from market_agent.concepts.ontology import TradingConcept
from market_agent.llm.interpreter import LLMNotConfiguredError
from market_agent.methodology.extractor import (
    LLMMethodologyExtractor, RuleBasedMethodologyExtractor,
)
from market_agent.methodology.schema import RawMethodologySource


def _source(text):
    return RawMethodologySource(name="Test System", practitioner="Test Trader", source_type="book",
                                 raw_text=text)


def test_rule_based_extractor_matches_breakout_and_volume():
    claims = RuleBasedMethodologyExtractor().extract(
        _source("A breakout above the range high on a volume surge confirms the setup."))
    concepts = {c.concept for c in claims}
    assert TradingConcept.BREAKOUT in concepts
    assert TradingConcept.VOLUME in concepts


def test_rule_based_extractor_matches_multiple_independent_concepts():
    claims = RuleBasedMethodologyExtractor().extract(
        _source("Look for a pullback to the moving average with relative strength versus the market, "
                "confirmed on multiple timeframes, using strict stop-loss position sizing."))
    concepts = {c.concept for c in claims}
    assert TradingConcept.PULLBACK in concepts
    assert TradingConcept.TREND in concepts
    assert TradingConcept.RELATIVE_STRENGTH in concepts
    assert TradingConcept.MULTI_TIMEFRAME_CONFIRMATION in concepts
    assert TradingConcept.RISK_MANAGEMENT in concepts


def test_rule_based_extractor_returns_empty_for_unrelated_text():
    claims = RuleBasedMethodologyExtractor().extract(
        _source("The company announced a new product line at its annual conference."))
    assert claims == []


def test_rule_based_extractor_rationale_never_asserts_profitability():
    claims = RuleBasedMethodologyExtractor().extract(_source("A classic breakout system."))
    assert len(claims) >= 1
    for c in claims:
        assert "win rate" not in c.rationale.lower()
        assert "profit" not in c.rationale.lower()
        assert "not evidence" in c.rationale.lower()


def test_rule_based_extractor_is_marked_as_narrow_coverage():
    assert RuleBasedMethodologyExtractor().coverage_is_narrow is True


def test_llm_extractor_construction_never_raises():
    """Selection alone must not fail - same contract as
    select_hypothesis_generator_from_env(): only calling .extract() raises."""
    extractor = LLMMethodologyExtractor()
    assert extractor.NAME == "LLM"


def test_llm_extractor_raises_not_configured_without_a_client():
    extractor = LLMMethodologyExtractor()
    with pytest.raises(LLMNotConfiguredError):
        extractor.extract(_source("anything"))


def test_llm_extractor_raises_not_implemented_with_a_client():
    extractor = LLMMethodologyExtractor(client=object())
    with pytest.raises(NotImplementedError):
        extractor.extract(_source("anything"))
