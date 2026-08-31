"""METHODOLOGY_EXTRACTOR_PROVIDER environment-variable selection - mirrors
tests/test_llm_select_env.py's coverage exactly."""
import pytest

from market_agent.llm.interpreter import LLMNotConfiguredError
from market_agent.methodology.select import describe_active_extractor, select_methodology_extractor_from_env


def test_default_is_rule_based_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("METHODOLOGY_EXTRACTOR_PROVIDER", raising=False)
    extractor = select_methodology_extractor_from_env()
    assert extractor.NAME == "RULE_BASED"


def test_llm_provider_without_client_raises_on_use_not_silently_falls_back(monkeypatch):
    monkeypatch.setenv("METHODOLOGY_EXTRACTOR_PROVIDER", "llm")
    extractor = select_methodology_extractor_from_env()  # selection itself doesn't raise
    assert extractor.NAME == "LLM"
    from market_agent.methodology.schema import RawMethodologySource
    with pytest.raises(LLMNotConfiguredError):
        extractor.extract(RawMethodologySource("X", "Y", "book", "text"))


def test_invalid_provider_value_raises():
    import os
    os.environ["METHODOLOGY_EXTRACTOR_PROVIDER"] = "not_a_real_provider"
    try:
        with pytest.raises(ValueError):
            select_methodology_extractor_from_env()
    finally:
        del os.environ["METHODOLOGY_EXTRACTOR_PROVIDER"]


def test_describe_active_extractor_flags_narrow_coverage(monkeypatch):
    monkeypatch.delenv("METHODOLOGY_EXTRACTOR_PROVIDER", raising=False)
    extractor = select_methodology_extractor_from_env()
    description = describe_active_extractor(extractor)
    assert "RULE_BASED" in description
    assert "narrow" in description.lower()
