"""Event interpretation: RawItem -> EventRecord.

Blueprint section D assigns "understand what happened" to the LLM -
entity resolution, narrative context, nuanced language. This module
deliberately does NOT do that yet: there is no LLM wired into this
execution environment, and fabricating a call I can't verify runs would
be worse than being honest about the gap. Instead, this defines the
`Interpreter` interface the rest of the system depends on, and ships one
concrete, rule-based implementation - the same keyword/pattern-matching
approach the deleted project's analysis/news_intelligence.py used
successfully for structured event types. An LLM-backed Interpreter drops
in later as a second implementation of the same interface; nothing
downstream (retrieval, prediction, logging) needs to change when it does.

Scoped to TWO event types as of stage 5 item 3 - guidance changes (stage 1)
and dividend changes (stage 5's first expansion). Both stay deliberately
narrow, structured, and low-ambiguity - exactly the profile that makes
rule-based extraction defensible without an LLM, and guidance is the same
event family EXP_010 already proved has clean, extractable signal at real
scale in the deleted project's prior research. Dividend changes were
chosen as the next type (over earnings surprises or analyst actions)
specifically because they need NO consensus-estimate or analyst-feed data
source this system doesn't have (see events/schema.py's ContextSnapshot
docstring for that disclosed gap) - a dividend increase/cut is a fact
about the filing itself, sourceable with the exact same proven EDGAR
full-text-search pattern as guidance (see sources/edgar_dividend.py).

GUIDANCE_CHANGE stays fully operational, unchanged - its patterns are
checked first, exactly as before this stage; DIVIDEND_CHANGE is a pure
addition, checked only once guidance doesn't match.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from market_agent.events.schema import ContextSnapshot, EventRecord, RawItem

GUIDANCE_RAISE_PATTERNS = [
    r"\braises?\s+(?:full[- ]year\s+)?guidance\b", r"\braises?\s+(?:full[- ]year\s+)?(?:forecast|outlook)\b",
    r"\bboosts?\s+guidance\b", r"\blifts?\s+(?:full[- ]year\s+)?(?:guidance|forecast|outlook)\b",
]
GUIDANCE_CUT_PATTERNS = [
    r"\bcuts?\s+(?:full[- ]year\s+)?guidance\b", r"\bcuts?\s+(?:full[- ]year\s+)?(?:forecast|outlook)\b",
    r"\blowers?\s+(?:full[- ]year\s+)?(?:guidance|forecast|outlook)\b",
    r"\bslashes?\s+(?:full[- ]year\s+)?(?:guidance|forecast|outlook)\b", r"\bwithdraws?\s+guidance\b",
]

# Verified live against efts.sec.gov 2018-2024 8-K full-text search before being hardcoded here (see
# sources/edgar_dividend.py's module docstring for the exact hit counts) - same discipline as
# GUIDANCE_RAISE_PATTERNS/GUIDANCE_CUT_PATTERNS above, and the same real, disclosed asymmetry: far
# more raw hits for dividend increases than cuts/suspensions (companies rarely use a sharp, formulaic
# verb near "dividend" when cutting one - "reduces", vague earnings-call language, or silence, as
# opposed to "increases quarterly dividend"'s formulaic upside phrasing).
DIVIDEND_RAISE_PATTERNS = [
    r"\bincreases?\s+(?:its\s+)?(?:quarterly\s+)?dividend\b", r"\braises?\s+(?:its\s+)?(?:quarterly\s+)?dividend\b",
    r"\bdeclares?\s+(?:a\s+)?special\s+dividend\b",
]
DIVIDEND_CUT_PATTERNS = [
    r"\bsuspends?\s+(?:its\s+|the\s+)?(?:quarterly\s+)?dividend\b",
    r"\beliminates?\s+(?:its\s+)?(?:quarterly\s+)?dividend\b",
    r"\bwill\s+not\s+pay\s+a\s+dividend\b", r"\breduces?\s+the\s+quarterly\s+dividend\b",
]

EVENT_TYPE_GUIDANCE_CHANGE = "GUIDANCE_CHANGE"
EVENT_TYPE_DIVIDEND_CHANGE = "DIVIDEND_CHANGE"


class Interpreter(ABC):
    """Anything that turns a RawItem into an EventRecord (or None, if the
    item doesn't represent a recognizable, in-scope event type).

    NAME is a required class attribute, not decoration: every concrete
    Interpreter must declare it, and callers (see market_agent/llm's
    factory functions) are expected to log/report it so it's always
    obvious, in any run's output, whether interpretation was done by the
    rule-based extractor or an LLM - never silently one when the other
    was claimed."""
    NAME: str

    @abstractmethod
    def interpret(self, item: RawItem, context: ContextSnapshot,
                  source_reliability_snapshot: float | None) -> EventRecord | None:
        raise NotImplementedError


@dataclass
class RuleBasedInterpreter(Interpreter):
    """Deterministic, auditable, keyword-pattern extraction. Every
    classification traces to the specific pattern that matched - no
    hidden judgment call, which is exactly what makes it safe to run
    without a human or LLM in the loop for Stage 1."""

    NAME = "RULE_BASED"

    def interpret(self, item: RawItem, context: ContextSnapshot,
                  source_reliability_snapshot: float | None) -> EventRecord | None:
        text = item.text.lower()

        direction, event_type = None, None
        if any(re.search(p, text) for p in GUIDANCE_RAISE_PATTERNS):
            direction, event_type = "positive", EVENT_TYPE_GUIDANCE_CHANGE
        elif any(re.search(p, text) for p in GUIDANCE_CUT_PATTERNS):
            direction, event_type = "negative", EVENT_TYPE_GUIDANCE_CHANGE
        elif any(re.search(p, text) for p in DIVIDEND_RAISE_PATTERNS):
            direction, event_type = "positive", EVENT_TYPE_DIVIDEND_CHANGE
        elif any(re.search(p, text) for p in DIVIDEND_CUT_PATTERNS):
            direction, event_type = "negative", EVENT_TYPE_DIVIDEND_CHANGE
        if direction is None:
            return None  # not a recognizable in-scope event - correctly out of scope, not guessed at

        # ingested_at = published_at: Stage 1 simplification (real-time ingestion assumed). A future
        # stage with a genuine ingestion delay (crawl lag, batch processing) should pass the actual
        # ingestion timestamp separately rather than conflating the two.
        return EventRecord(
            entity=item.entity, event_type=event_type, direction=direction,
            source=item.source, source_reliability_snapshot=source_reliability_snapshot,
            raw_text=item.text, published_at=item.published_at, ingested_at=item.published_at,
            context=context.to_dict(),
        )
