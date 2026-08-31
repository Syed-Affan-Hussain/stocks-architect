"""Validates the negation handling added to classify_sentiment - grounded
in SENTiVENT (Jacobs & Hoste; see extraction.py's NEGATION_CUES docstring),
which annotates negation as an explicit event attribute distinct from
polarity itself, precisely because "revenue did not decline" and "revenue
declined" describe opposite realities despite sharing the word "decline".

Also locks in the deliberate scope limits so a future change doesn't
silently widen or narrow negation detection without a test noticing:
bare "no" is excluded (idiom false-positives: "no doubt", "no signs of
slowing"), and a clause with genuinely mixed polarity is left as MIXED
rather than guessed at (no negation-scope parser is available)."""
from market_agent.research.extraction import classify_sentiment


def test_negated_negative_word_reads_positive():
    assert classify_sentiment("Revenue did not decline this quarter for the company") == "POSITIVE"


def test_negated_positive_word_reads_negative():
    assert classify_sentiment("The company said margins will not expand this year") == "NEGATIVE"


def test_unnegated_negative_clause_still_reads_negative():
    """The negation logic must not over-fire on ordinary negative clauses
    that happen to share no negation cue."""
    assert classify_sentiment("Revenue declined sharply this quarter for the company") == "NEGATIVE"


def test_contraction_negation_is_detected():
    assert classify_sentiment("Margins won't decline further, the company said") == "POSITIVE"


def test_denied_flips_negative_claim():
    assert classify_sentiment("The company denied plans to cut costs") == "POSITIVE"


def test_bare_no_is_not_treated_as_negation_idiom_guard():
    """"no doubt" / "no signs of slowing" are common financial-reporting
    idioms that use "no" as an intensifier, not a negator - bare "no" is
    deliberately excluded from NEGATION_CUES so this stays POSITIVE
    instead of incorrectly flipping to NEGATIVE."""
    assert classify_sentiment("There is no doubt demand remains strong") == "POSITIVE"


def test_genuinely_mixed_clause_is_unaffected_by_absent_negation():
    assert classify_sentiment("Revenue increased but margins declined") == "MIXED"


def test_negation_without_any_polarity_word_stays_neutral():
    """Negation alone flips an existing polarity read - it doesn't
    manufacture one where the lexicon found nothing to flip."""
    assert classify_sentiment("The company did not comment on the matter") == "NEUTRAL"
