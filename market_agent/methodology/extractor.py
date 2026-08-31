"""Methodology -> canonical-concept extraction (stage 6). Same
rule-based-stand-in / LLM-slot split as events/interpret.py and
learn/hypothesis.py, for the same reason: there is no LLM SDK/API key
configured in this execution environment (verified repeatedly across this
project's whole history - see llm/interpreter.py's module docstring), and
fabricating a call that can't be run/tested would be worse than being
honest about the gap.

WHY RULE-BASED EXTRACTION HERE IS GENUINELY NARROW, MORE SO THAN
events/interpret.py's: guidance/dividend text is short, formulaic
newswire language with a handful of standard verb phrases. Free-form
trading-methodology prose (a book chapter, an interview transcript) is
not - mapping arbitrary long-form text onto a 20-concept ontology with any
generality is fundamentally a semantic-understanding task, i.e. exactly
what an LLM is for. RuleBasedMethodologyExtractor below only recognizes a
small, fixed set of standard technical-analysis phrases/keywords - it
works on the kind of already-fairly-structured, keyword-dense description
this project's own seed_corpus.py writes (deliberately, so the rule-based
path has something real to run against), but it will silently miss
genuine concept content in truly free-form prose. This is disclosed, not
hidden: `coverage_is_narrow = True` is a class attribute for exactly this
reason, and the knowledge-state report surfaces which extractor produced
each methodology's links.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from market_agent.concepts.ontology import TradingConcept
from market_agent.methodology.schema import ExtractedConceptClaim, RawMethodologySource

# Curated, fixed phrase/keyword patterns per concept - standard technical-analysis vocabulary, not
# tuned against any backtest result. Deliberately covers all 20 ontology concepts (including the 3
# not computable by technical_context.py - OPENING_RANGE, SECTOR_CONTEXT, RISK_MANAGEMENT) because a
# methodology CAN legitimately claim to use one of those; this system just can't test a hypothesis
# conditioned on it later (see learn/hypothesis.py's CONDITIONING_DIMENSIONS, which only draws from
# the computable 17).
CONCEPT_PATTERNS: dict[TradingConcept, list[str]] = {
    TradingConcept.TREND: [r"\btrend[- ]?following\b", r"\buptrends?\b", r"\bdowntrends?\b",
                            r"\bmoving averages?\b", r"\btrend template\b"],
    TradingConcept.MOMENTUM: [r"\bmomentum\b", r"\brate of change\b", r"\baccelerating\b"],
    TradingConcept.MEAN_REVERSION: [r"\bmean reversion\b", r"\brevert(?:s|ing)? to the average\b",
                                     r"\boversold\b", r"\boverbought\b"],
    TradingConcept.BREAKOUT: [r"\bbreakouts?\b", r"\bbreaks? out\b", r"\bclears? (?:the |a )?(?:prior )?high\b"],
    TradingConcept.PULLBACK: [r"\bpullbacks?\b", r"\bpulls? back\b", r"\bretracements?\b", r"\bdip buy(?:ing|s)?\b"],
    TradingConcept.PRICE_ACTION: [r"\bengulfing\b", r"\binside bars?\b", r"\boutside bars?\b",
                                   r"\bcandlestick\b", r"\bprice action\b"],
    TradingConcept.SUPPORT_RESISTANCE: [r"\bsupport\b", r"\bresistance\b", r"\bprior highs?\b", r"\bprior lows?\b"],
    TradingConcept.VOLATILITY_COMPRESSION_EXPANSION: [r"\bvolatility contraction\b", r"\bvolatility squeeze\b",
                                                        r"\btight(?:ening)? range\b", r"\bexpanding range\b"],
    TradingConcept.VOLUME: [r"\bvolume surge\b", r"\bheavy volume\b", r"\bvolume spike\b", r"\bhigh volume\b"],
    TradingConcept.RELATIVE_VOLUME: [r"\brelative volume\b", r"\bvolume relative to (?:its )?average\b"],
    TradingConcept.VWAP: [r"\bvwap\b", r"\bvolume[- ]weighted average price\b"],
    TradingConcept.MOVING_AVERAGE_STRUCTURE: [r"\bmoving average stack\b", r"\b20.?50.?200\b", r"\bstage 2\b"],
    TradingConcept.MARKET_STRUCTURE: [r"\bhigher highs\b", r"\bhigher lows\b", r"\blower highs\b",
                                       r"\blower lows\b", r"\bmarket structure\b"],
    TradingConcept.GAPS: [r"\bgaps? up\b", r"\bgaps? down\b", r"\bgap and go\b"],
    TradingConcept.OPENING_RANGE: [r"\bopening range\b", r"\bfirst (?:five|5|fifteen|15|thirty|30) minutes\b",
                                    r"\borb\b"],
    TradingConcept.RELATIVE_STRENGTH: [r"\brelative strength\b", r"\boutperform(?:ing|s)? the market\b",
                                        r"\bcanslim\b"],
    TradingConcept.SECTOR_CONTEXT: [r"\bsector rotation\b", r"\bsector strength\b", r"\bsector leadership\b",
                                     r"\bindustry group\b"],
    TradingConcept.MULTI_TIMEFRAME_CONFIRMATION: [r"\bmultiple time ?frames?\b", r"\bweekly confirmation\b",
                                                    r"\bdaily and weekly\b"],
    TradingConcept.CATALYST_EVENT_REACTION: [r"\bearnings reaction\b", r"\bcatalyst\b", r"\bnews[- ]driven\b"],
    TradingConcept.RISK_MANAGEMENT: [r"\bstop[- ]loss\b", r"\bposition siz(?:e|ing)\b",
                                      r"\brisk (?:1|one) percent\b", r"\brisk[:/]reward\b"],
}


class MethodologyExtractor(ABC):
    NAME: str
    coverage_is_narrow: bool = False

    @abstractmethod
    def extract(self, source: RawMethodologySource) -> list[ExtractedConceptClaim]:
        raise NotImplementedError


class RuleBasedMethodologyExtractor(MethodologyExtractor):
    NAME = "RULE_BASED"
    coverage_is_narrow = True  # see module docstring - narrow, keyword-pattern coverage only

    def extract(self, source: RawMethodologySource) -> list[ExtractedConceptClaim]:
        text = source.raw_text.lower()
        claims = []
        for concept, patterns in CONCEPT_PATTERNS.items():
            matched = [p for p in patterns if re.search(p, text)]
            if matched:
                claims.append(ExtractedConceptClaim(
                    concept=concept,
                    rationale=f"Rule-based match on {matched[0]!r} in {source.name}'s description "
                              f"(source: {source.practitioner}, {source.source_type}). Paraphrased, "
                              "audit-trail only - not evidence this concept predicts anything."))
        return claims


class LLMMethodologyExtractor(MethodologyExtractor):
    """Real implementation intentionally NOT WRITTEN - see this module's
    docstring and llm/interpreter.py's LLMInterpreter for the identical
    reasoning and the identical construction contract: constructing this
    class WITHOUT a client never raises (selection alone must not fail -
    see llm/select.py's select_hypothesis_generator_from_env, which this
    mirrors); only calling .extract() raises, LLMNotConfiguredError with
    no client or NotImplementedError with one, since the actual
    prompt/parsing logic was never written against a real, testable SDK."""
    NAME = "LLM"

    def __init__(self, client=None):
        self.client = client

    def extract(self, source: RawMethodologySource) -> list[ExtractedConceptClaim]:
        if self.client is None:
            from market_agent.llm.interpreter import LLMNotConfiguredError
            raise LLMNotConfiguredError(
                "LLMMethodologyExtractor has no client configured - this is the interface only. Use "
                "market_agent.methodology.extractor.RuleBasedMethodologyExtractor directly, and log that "
                "choice (see methodology/select.py's describe_active_extractor), if you want a working "
                "extractor today.")
        raise NotImplementedError(
            "LLMMethodologyExtractor.extract is not implemented - no LLM SDK/API key is configured in this "
            "environment (checked directly, not assumed). A real implementation would prompt the client with "
            "the ontology's 20 concept definitions and source.raw_text, constrained to return only concepts "
            "from that fixed list plus a short rationale - never a free-form new concept.")
