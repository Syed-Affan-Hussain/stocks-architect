from market_agent.learn.error_taxonomy import classify_error


def test_correct_direction_and_magnitude_is_ok():
    c = classify_error(predicted_impact=-0.04, predicted_confidence="MEDIUM",
                        realized_abnormal_return=-0.045, outcome_status="OK")
    assert c.error_type == "OK"
    assert c.may_learn_from is False


def test_opposite_sign_is_wrong_direction():
    c = classify_error(predicted_impact=-0.05, predicted_confidence="MEDIUM",
                        realized_abnormal_return=0.06, outcome_status="OK")
    assert c.error_type == "WRONG_DIRECTION"
    assert c.may_learn_from is True


def test_same_direction_large_magnitude_miss_is_wrong_magnitude():
    c = classify_error(predicted_impact=-0.01, predicted_confidence="MEDIUM",
                        realized_abnormal_return=-0.08, outcome_status="OK")
    assert c.error_type == "WRONG_MAGNITUDE"
    assert c.may_learn_from is True


def test_missing_outcome_is_data_error_and_never_learnable():
    c = classify_error(predicted_impact=-0.02, predicted_confidence="MEDIUM",
                        realized_abnormal_return=None, outcome_status="INSUFFICIENT_DATA")
    assert c.error_type == "DATA_ERROR"
    assert c.may_learn_from is False


def test_no_prediction_made_is_insufficient_data_and_never_learnable():
    c = classify_error(predicted_impact=None, predicted_confidence="INSUFFICIENT_PRECEDENT",
                        realized_abnormal_return=0.05, outcome_status="OK")
    assert c.error_type == "INSUFFICIENT_DATA"
    assert c.may_learn_from is False


def test_tiny_realized_move_does_not_trigger_wrong_direction():
    """A predicted -3% vs. a realized +0.1% (noise-level) should not be scored as a
    direction miss - both are effectively 'flat', per DIRECTION_MATTERS_THRESHOLD."""
    c = classify_error(predicted_impact=-0.03, predicted_confidence="MEDIUM",
                        realized_abnormal_return=0.001, outcome_status="OK")
    assert c.error_type != "WRONG_DIRECTION"
