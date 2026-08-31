from market_agent.research.evaluation.decision_mapping import (
    ASSESSMENT_TO_SIGNED_IMPACT, confidence_float_to_bucket, map_assessment_to_decision, news_state_to_decision,
)
from market_agent.research.schema import ASSESSMENTS


def test_every_assessment_label_has_a_mapping():
    assert set(ASSESSMENT_TO_SIGNED_IMPACT) == set(ASSESSMENTS)


def test_favorable_maps_to_positive_one():
    d = map_assessment_to_decision("FAVORABLE", 0.8)
    assert d.predicted_impact == 1.0
    assert d.predicted_confidence == 0.8


def test_negative_maps_to_negative_one():
    d = map_assessment_to_decision("NEGATIVE", 0.6)
    assert d.predicted_impact == -1.0


def test_neutral_and_uncertain_both_map_to_zero():
    assert map_assessment_to_decision("NEUTRAL", 0.5).predicted_impact == 0.0
    assert map_assessment_to_decision("UNCERTAIN", 0.5).predicted_impact == 0.0


def test_insufficient_evidence_maps_to_no_signal_not_zero():
    d = map_assessment_to_decision("INSUFFICIENT_EVIDENCE", None)
    assert d.predicted_impact is None
    assert d.predicted_confidence is None  # confidence dropped too - no signal means no signal


def test_scale_is_monotonic_favorable_to_negative():
    order = ["FAVORABLE", "CAUTIOUSLY_FAVORABLE", "NEUTRAL", "CAUTIOUS", "NEGATIVE"]
    values = [map_assessment_to_decision(a, 0.5).predicted_impact for a in order]
    assert values == sorted(values, reverse=True)


def test_unknown_assessment_label_raises_rather_than_silently_defaulting():
    import pytest
    with pytest.raises(ValueError):
        map_assessment_to_decision("SOMETHING_MADE_UP", 0.5)


def test_confidence_bucket_thresholds():
    assert confidence_float_to_bucket(0.9) == "HIGH"
    assert confidence_float_to_bucket(0.5) == "MEDIUM"
    assert confidence_float_to_bucket(0.1) == "LOW"
    assert confidence_float_to_bucket(None) == "LOW"


def test_news_state_none_yields_no_signal():
    d = news_state_to_decision(None)
    assert d.predicted_impact is None
    assert d.decision_label == "NEWS_UNAVAILABLE"


def test_news_state_with_no_populated_axes_yields_no_signal():
    d = news_state_to_decision({"dimensions": {"growth": None, "demand": None}, "confidence": 0.3})
    assert d.predicted_impact is None
    assert d.decision_label == "NEWS_NO_SIGNAL"


def test_news_state_mean_of_populated_axes_is_the_signal():
    d = news_state_to_decision({"dimensions": {"growth": 1.0, "demand": 0.5, "risk": None}, "confidence": 0.4})
    assert d.predicted_impact == 0.75
    assert d.decision_label == "NEWS_ONLY_UP"
    assert d.predicted_confidence == 0.4


def test_news_state_negative_mean_labels_down():
    d = news_state_to_decision({"dimensions": {"growth": -0.5, "risk": -0.5}, "confidence": 0.4})
    assert d.predicted_impact == -0.5
    assert d.decision_label == "NEWS_ONLY_DOWN"
