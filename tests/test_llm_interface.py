"""Proves the core safety property of the LLM interface layer: asking for
LLM behavior without a configured client fails loudly, and never silently
returns rule-based results while claiming otherwise."""
import pytest

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.events.schema import ContextSnapshot, RawItem
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.llm.hypothesis_generator import LLMHypothesisGenerator
from market_agent.llm.interpreter import LLMInterpreter, LLMNotConfiguredError
from market_agent.llm.select import describe_active_choice, select_hypothesis_generator, select_interpreter

CTX = ContextSnapshot(regime="NORMAL", prior_5d_return=0.0, sector_momentum="NEUTRAL")


def test_rule_based_interpreter_has_a_name():
    assert RuleBasedInterpreter.NAME == "RULE_BASED"


def test_llm_interpreter_has_a_name():
    assert LLMInterpreter.NAME == "LLM"


def test_unconfigured_llm_interpreter_raises_not_silently_falls_back():
    interpreter = LLMInterpreter(client=None)
    item = RawItem(text="Acme Corp cuts guidance", source="wire", entity="ACME",
                    published_at=__import__("datetime").datetime(2024, 1, 1))
    with pytest.raises(LLMNotConfiguredError):
        interpreter.interpret(item, CTX, 0.5)


def test_unconfigured_llm_hypothesis_generator_raises():
    generator = LLMHypothesisGenerator(client=None)
    with pytest.raises(LLMNotConfiguredError):
        generator.generate({"event_type": "GUIDANCE_CHANGE"}, "WRONG_DIRECTION")


def test_select_interpreter_returns_rule_based_by_default():
    interpreter = select_interpreter(use_llm=False)
    assert interpreter.NAME == "RULE_BASED"
    assert isinstance(interpreter, RuleBasedInterpreter)


def test_select_interpreter_with_llm_and_no_client_raises_immediately():
    with pytest.raises(LLMNotConfiguredError):
        select_interpreter(use_llm=True).interpret(
            RawItem("x", "wire", "AAPL", __import__("datetime").datetime(2024, 1, 1)), CTX, 0.5)


def test_describe_active_choice_flags_rule_based_explicitly():
    description = describe_active_choice(RuleBasedInterpreter(), RuleBasedHypothesisGenerator())
    assert "RULE_BASED" in description
    assert "NO LLM reasoning" in description


def test_describe_active_choice_does_not_flag_llm_as_rule_based():
    description = describe_active_choice(LLMInterpreter(client=object()), LLMHypothesisGenerator(client=object()))
    assert "NO LLM reasoning" not in description
    assert "LLM" in description
