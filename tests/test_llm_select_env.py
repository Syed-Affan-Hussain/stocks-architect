"""Stage 5: HYPOTHESIS_PROVIDER/INTERPRETER_PROVIDER environment-variable
selection - proves the default is rule-based, an explicit 'llm' request
without a client raises rather than silently falling back, and an
invalid value fails loudly rather than being coerced to a default."""
import pytest

from market_agent.llm.interpreter import LLMNotConfiguredError
from market_agent.llm.select import select_hypothesis_generator_from_env, select_interpreter_from_env


def test_default_is_rule_based_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("INTERPRETER_PROVIDER", raising=False)
    monkeypatch.delenv("HYPOTHESIS_PROVIDER", raising=False)
    assert select_interpreter_from_env().NAME == "RULE_BASED"
    assert select_hypothesis_generator_from_env().NAME == "RULE_BASED"


def test_explicit_rule_based_is_honored(monkeypatch):
    monkeypatch.setenv("HYPOTHESIS_PROVIDER", "rule_based")
    assert select_hypothesis_generator_from_env().NAME == "RULE_BASED"


def test_llm_provider_without_client_raises_on_use_not_silently_falls_back(monkeypatch):
    monkeypatch.setenv("HYPOTHESIS_PROVIDER", "llm")
    generator = select_hypothesis_generator_from_env()  # selection itself doesn't raise
    assert generator.NAME == "LLM"
    with pytest.raises(LLMNotConfiguredError):
        generator.generate({"event_type": "GUIDANCE_CHANGE"}, "WRONG_DIRECTION")


def test_invalid_provider_value_raises():
    import os
    os.environ["HYPOTHESIS_PROVIDER"] = "gpt5_please"
    try:
        with pytest.raises(ValueError):
            select_hypothesis_generator_from_env()
    finally:
        del os.environ["HYPOTHESIS_PROVIDER"]


def test_interpreter_and_hypothesis_provider_are_independently_settable(monkeypatch):
    monkeypatch.setenv("INTERPRETER_PROVIDER", "rule_based")
    monkeypatch.setenv("HYPOTHESIS_PROVIDER", "llm")
    assert select_interpreter_from_env().NAME == "RULE_BASED"
    assert select_hypothesis_generator_from_env().NAME == "LLM"
