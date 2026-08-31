"""LLM-backed Interpreter - interface defined, provider deliberately left
unimplemented. Do not pretend an LLM is available in this execution
environment; do not silently substitute the rule-based interpreter while
this class's NAME claims "LLM" was used.

WHAT WOULD BE NEEDED TO IMPLEMENT THIS FOR REAL: a `client` satisfying
`LLMClient` below - one method, `complete_structured(prompt, schema) ->
dict`, matching the schema-validated / function-calling style already
established as the requirement in the Adaptive Market-Intelligence
Blueprint (§16: "every LLM call returns a typed, schema-validated
object... never raw prose feeding downstream logic"). Wiring in a real
provider means: (1) constructing a concrete LLMClient (e.g. an Anthropic
Messages API client using tool-use / structured output), (2) building a
prompt from RawItem+ContextSnapshot that asks specifically for
{event_type, direction} given this stage's fixed, narrow taxonomy (do not
widen the event-type vocabulary here - that's a later stage's decision,
not this interpreter's), (3) validating the structured response against
EventRecord's fields before constructing one - a malformed/unparseable
response must raise, never be guessed at.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from market_agent.events.interpret import Interpreter
from market_agent.events.schema import ContextSnapshot, EventRecord, RawItem


class LLMNotConfiguredError(Exception):
    """Raised whenever LLMInterpreter/LLMHypothesisGenerator is invoked
    without a real client. This must never be caught and silently routed
    to the rule-based implementation - if that substitution is wanted,
    the CALLER makes that choice explicitly (see market_agent/llm/select.py),
    logs which implementation is actually running, and never claims LLM
    reasoning occurred when it didn't."""


class LLMClient(ABC):
    """The minimal contract any real provider integration must satisfy.
    Deliberately abstract here - no network code, no API key handling,
    no vendor SDK import - this module defines the shape of the
    dependency, not the dependency itself."""

    @abstractmethod
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Returns a dict already validated against `schema`. A real
        implementation should raise on a response that doesn't validate,
        never coerce/guess a best-effort value."""
        raise NotImplementedError


@dataclass
class LLMInterpreter(Interpreter):
    NAME = "LLM"

    client: LLMClient | None = None

    def interpret(self, item: RawItem, context: ContextSnapshot,
                  source_reliability_snapshot: float | None) -> EventRecord | None:
        if self.client is None:
            raise LLMNotConfiguredError(
                "LLMInterpreter has no client configured - this is the interface only. See this "
                "module's docstring for what a real implementation needs. Use "
                "market_agent.events.interpret.RuleBasedInterpreter directly, and log that choice, "
                "if you want a working interpreter today.")
        # A real implementation would build a prompt from item/context, call
        # self.client.complete_structured(...) with an {event_type, direction} schema restricted to
        # this stage's fixed taxonomy (EVENT_TYPE_GUIDANCE_CHANGE only - see module docstring), and
        # construct an EventRecord from the validated result. Not implemented - see class docstring.
        raise NotImplementedError("LLMInterpreter.interpret is not implemented - see module docstring.")
