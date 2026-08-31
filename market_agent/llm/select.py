"""The one place a pipeline decides between the rule-based and LLM-backed
implementations of Interpreter/HypothesisGenerator - so that decision is
never buried inside a try/except fallback somewhere, and every run can
report which one it actually used.

There is no "auto-detect and silently fall back" mode here on purpose.
Requesting the LLM implementation without a configured client raises
immediately (LLMNotConfiguredError) rather than quietly returning the
rule-based one - see market_agent/llm/interpreter.py's docstring for why
that distinction matters. A caller who wants graceful degradation must
catch that error explicitly and decide for itself, at the call site,
whether to proceed with the rule-based implementation - and if it does,
it must log that decision (see `describe_active_choice`), never present
the resulting predictions as LLM-reasoned.
"""
from __future__ import annotations

import os

from market_agent.events.interpret import Interpreter, RuleBasedInterpreter
from market_agent.learn.hypothesis import HypothesisGenerator, RuleBasedHypothesisGenerator
from market_agent.llm.hypothesis_generator import LLMHypothesisGenerator
from market_agent.llm.interpreter import LLMClient, LLMInterpreter

# Stage 5: explicit provider configuration via environment variable, per this stage's own
# requirement ("HYPOTHESIS_PROVIDER=rule_based" / "HYPOTHESIS_PROVIDER=llm"). Defaults to
# rule_based - an unset environment must never silently mean "try the LLM", and an explicit
# "llm" request with no client configured must fail loudly (LLMNotConfiguredError), never
# silently fall back. INTERPRETER_PROVIDER mirrors it for the interpretation stage, independently
# settable since a deployment could reasonably want one LLM-backed and not the other.
VALID_PROVIDERS = ("rule_based", "llm")


def select_interpreter(use_llm: bool, client: LLMClient | None = None) -> Interpreter:
    if use_llm:
        return LLMInterpreter(client=client)  # raises LLMNotConfiguredError at call time if client is None
    return RuleBasedInterpreter()


def select_hypothesis_generator(use_llm: bool, client: LLMClient | None = None) -> HypothesisGenerator:
    if use_llm:
        return LLMHypothesisGenerator(client=client)
    return RuleBasedHypothesisGenerator()


def describe_active_choice(interpreter: Interpreter, hypothesis_generator: HypothesisGenerator) -> str:
    """A one-line, unambiguous statement of what's actually running -
    intended to be printed/logged at the start of every pipeline run
    (see experiment/walkforward.py) so it's never ambiguous after the
    fact whether a given run's predictions came from rule-based
    extraction or genuine LLM reasoning."""
    return (f"Interpreter: {interpreter.NAME}  |  Hypothesis generator: {hypothesis_generator.NAME}"
            + ("  <- rule-based stand-in, NO LLM reasoning occurred in this run"
               if interpreter.NAME == "RULE_BASED" else ""))


def _read_provider(env_var: str) -> str:
    value = os.environ.get(env_var, "rule_based").strip().lower()
    if value not in VALID_PROVIDERS:
        raise ValueError(f"{env_var}={value!r} is not a valid provider - must be one of {VALID_PROVIDERS}.")
    return value


def select_interpreter_from_env(client: LLMClient | None = None) -> Interpreter:
    """Reads INTERPRETER_PROVIDER (default 'rule_based'). 'llm' with no
    client configured raises LLMNotConfiguredError the first time
    .interpret() is actually called - selection itself never silently
    substitutes rule-based just because a client wasn't supplied."""
    return select_interpreter(_read_provider("INTERPRETER_PROVIDER") == "llm", client=client)


def select_hypothesis_generator_from_env(client: LLMClient | None = None) -> HypothesisGenerator:
    """Reads HYPOTHESIS_PROVIDER (default 'rule_based') - this stage's
    explicit requirement. Same no-silent-fallback semantics as
    select_interpreter_from_env."""
    return select_hypothesis_generator(_read_provider("HYPOTHESIS_PROVIDER") == "llm", client=client)
