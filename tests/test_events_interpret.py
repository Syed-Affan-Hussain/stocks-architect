from datetime import datetime, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.events.schema import ContextSnapshot, RawItem

CTX = ContextSnapshot(regime="NORMAL", prior_5d_return=0.0, sector_momentum="NEUTRAL")


def _item(text):
    return RawItem(text=text, source="test-wire", entity="NVDA", published_at=datetime(2024, 1, 1, tzinfo=timezone.utc))


def test_guidance_cut_classified_negative():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp cuts full-year guidance amid weak demand"), CTX, 0.6)
    assert event is not None
    assert event.event_type == "GUIDANCE_CHANGE"
    assert event.direction == "negative"


def test_guidance_raise_classified_positive():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp raises full-year guidance on strong bookings"), CTX, 0.6)
    assert event is not None
    assert event.direction == "positive"


def test_guidance_withdrawal_classified_negative():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp withdraws guidance citing uncertainty"), CTX, 0.6)
    assert event is not None
    assert event.direction == "negative"


def test_unrelated_headline_returns_none_not_a_guess():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp announces new CEO"), CTX, 0.6)
    assert event is None


def test_ambiguous_mention_of_guidance_without_a_verb_returns_none():
    event = RuleBasedInterpreter().interpret(_item("Analysts discuss Acme Corp guidance ahead of earnings"), CTX, 0.6)
    assert event is None


def test_source_reliability_and_context_carried_through():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp cuts guidance"), CTX, 0.42)
    assert event.source_reliability_snapshot == 0.42
    assert event.context["regime"] == "NORMAL"


def test_dividend_increase_classified_positive():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp increases quarterly dividend by 10%"), CTX, 0.6)
    assert event is not None
    assert event.event_type == "DIVIDEND_CHANGE"
    assert event.direction == "positive"


def test_special_dividend_classified_positive():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp declares special dividend of $1.00 per share"),
                                              CTX, 0.6)
    assert event is not None
    assert event.event_type == "DIVIDEND_CHANGE"
    assert event.direction == "positive"


def test_dividend_suspension_classified_negative():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp suspends its quarterly dividend amid cash crunch"),
                                              CTX, 0.6)
    assert event is not None
    assert event.event_type == "DIVIDEND_CHANGE"
    assert event.direction == "negative"


def test_dividend_elimination_classified_negative():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp eliminates dividend to preserve cash"), CTX, 0.6)
    assert event is not None
    assert event.event_type == "DIVIDEND_CHANGE"
    assert event.direction == "negative"


def test_guidance_still_takes_priority_and_stays_operational():
    """Guidance patterns are checked first - a headline that happens to
    mention both should still classify as GUIDANCE_CHANGE, matching this
    stage's explicit requirement to keep guidance changes operational
    while adding the next type."""
    event = RuleBasedInterpreter().interpret(
        _item("Acme Corp cuts full-year guidance and increases quarterly dividend"), CTX, 0.6)
    assert event is not None
    assert event.event_type == "GUIDANCE_CHANGE"
    assert event.direction == "negative"


def test_unrelated_headline_still_returns_none_with_dividend_patterns_added():
    event = RuleBasedInterpreter().interpret(_item("Acme Corp announces new CEO"), CTX, 0.6)
    assert event is None
