"""LLM-backed HypothesisGenerator - same status as llm/interpreter.py:
interface defined, provider deliberately unimplemented. See that module's
docstring for the shared LLMClient contract and LLMNotConfiguredError
semantics (never silently substitute the rule-based generator).

A real implementation is where the blueprint's actual LLM value-add for
hypothesis generation lives (§G/§13): given a failed prediction's event,
the full expanded context (experiment/context.py - prior returns at
multiple horizons, realized vol, market return, event timing, recent
related events), and retrieved similar historical cases, propose
genuinely reasoned candidate explanations - richer ones than
RuleBasedHypothesisGenerator's two fixed shapes (regime alone; regime +
prior-return-bucket). Whatever it proposes still only ever produces a
list of ProposedHypothesis (each a structured condition + audit-trail
prose) that goes into learn/governance.py's SAME testing gate - this
class never gets to skip that gate just because an LLM produced the
hypotheses instead of a rule. Returning MULTIPLE candidates per error is
fine and expected (an LLM can reasonably propose several competing
explanations for one miss) - test_hypotheses_batch's Holm-Bonferroni
correction already accounts for however many are tested together,
whether they came from one generator call or many.
"""
from __future__ import annotations

from dataclasses import dataclass

from market_agent.llm.interpreter import LLMClient, LLMNotConfiguredError
from market_agent.learn.hypothesis import HypothesisGenerator, ProposedHypothesis


@dataclass
class LLMHypothesisGenerator(HypothesisGenerator):
    NAME = "LLM"

    client: LLMClient | None = None

    def generate(self, event_row, error_type: str) -> list[ProposedHypothesis]:
        if self.client is None:
            raise LLMNotConfiguredError(
                "LLMHypothesisGenerator has no client configured - this is the interface only. Use "
                "market_agent.learn.hypothesis.RuleBasedHypothesisGenerator directly, and log that "
                "choice, if you want a working generator today.")
        raise NotImplementedError("LLMHypothesisGenerator.generate is not implemented - see module docstring.")
