from market_agent.research.news_state.magnitude import (
    DIRECTION_ONLY_SCORE, explicit_sign, extract_magnitudes, extract_primary_magnitude, magnitude_to_score,
)


def test_extracts_percent():
    facts = extract_magnitudes("Revenue grew 40% year over year.")
    assert len(facts) == 1
    assert facts[0].unit == "PERCENT"
    assert facts[0].value == 40.0


def test_extracts_bps():
    facts = extract_magnitudes("Margins contracted 200 basis points.")
    assert facts[0].unit == "BPS"
    assert facts[0].value == 200.0


def test_extracts_dollar_with_billion_unit():
    facts = extract_magnitudes("The company raised guidance to $30 billion.")
    assert facts[0].unit == "USD"
    assert facts[0].value == 30_000_000_000.0


def test_extracts_dollar_with_million_unit():
    facts = extract_magnitudes("A $500 million buyback was announced.")
    assert facts[0].unit == "USD"
    assert facts[0].value == 500_000_000.0


def test_no_magnitude_returns_empty_list():
    assert extract_magnitudes("The company held its annual meeting.") == []


def test_primary_magnitude_prefers_percent_over_bps_and_usd():
    fact = extract_primary_magnitude("Revenue grew 40%, funded by a $2 billion investment.")
    assert fact.unit == "PERCENT"
    assert fact.value == 40.0


def test_primary_magnitude_none_when_nothing_found():
    assert extract_primary_magnitude("Nothing quantitative here.") is None


def test_magnitude_to_score_anchors_are_monotonic_and_bounded():
    small = magnitude_to_score(extract_primary_magnitude("grew 2%"))
    medium = magnitude_to_score(extract_primary_magnitude("grew 10%"))
    large = magnitude_to_score(extract_primary_magnitude("grew 40%"))
    huge = magnitude_to_score(extract_primary_magnitude("grew 90%"))
    assert 0 < small < medium < large <= 1.0
    assert large == huge == 1.0  # saturates beyond the 30% anchor - not distinguishable further


def test_magnitude_to_score_exact_anchor_values():
    assert magnitude_to_score(extract_primary_magnitude("grew 5%")) == 0.3
    assert magnitude_to_score(extract_primary_magnitude("grew 15%")) == 0.6
    assert magnitude_to_score(extract_primary_magnitude("grew 30%")) == 1.0
    assert magnitude_to_score(extract_primary_magnitude("grew 0%")) == 0.0


def test_magnitude_to_score_usd_is_not_scored():
    fact = extract_primary_magnitude("raised guidance to $30 billion")
    assert magnitude_to_score(fact) is None


def test_two_percent_magnitudes_produce_genuinely_different_scores():
    """The core fix this module exists for: a 2% move and a 40% move must
    NOT both collapse to the same +-1.0."""
    small = magnitude_to_score(extract_primary_magnitude("Revenue grew 2%."))
    large = magnitude_to_score(extract_primary_magnitude("Revenue grew 40%."))
    assert small != large
    assert small < 0.3 < large


def test_explicit_sign_detected_only_with_literal_minus():
    assert explicit_sign(extract_primary_magnitude("revenue changed -5%")) == -1
    assert explicit_sign(extract_primary_magnitude("revenue declined 5%")) is None  # no literal minus


def test_direction_only_score_sits_between_small_and_moderate_anchors():
    assert 0.3 < DIRECTION_ONLY_SCORE < 0.6
