import pytest

from market_agent.strategy.test_isolation import TestIsolationGuard, TestIsolationViolation


def test_parameter_selection_allowed_before_test_observed():
    guard = TestIsolationGuard()
    guard.assert_parameter_selection_allowed("construct StrategyAgent")  # must not raise


def test_parameter_selection_blocked_after_test_observed():
    guard = TestIsolationGuard()
    guard.mark_test_observed("printed TEST Sharpe ratio")
    with pytest.raises(TestIsolationViolation):
        guard.assert_parameter_selection_allowed("retune cost_margin_multiple")


def test_violation_message_names_the_action_and_first_observation_context():
    guard = TestIsolationGuard()
    guard.mark_test_observed("printed TEST Sharpe ratio")
    try:
        guard.assert_parameter_selection_allowed("retune cost_margin_multiple")
        assert False, "expected TestIsolationViolation"
    except TestIsolationViolation as e:
        assert "retune cost_margin_multiple" in str(e)
        assert "printed TEST Sharpe ratio" in str(e)


def test_mark_test_observed_is_idempotent_keeps_first_context():
    guard = TestIsolationGuard()
    guard.mark_test_observed("first read")
    guard.mark_test_observed("second read")
    assert guard.test_observed_context == "first read"


def test_reset_allows_a_fresh_run():
    guard = TestIsolationGuard()
    guard.mark_test_observed("printed TEST Sharpe ratio")
    guard.reset()
    assert guard.test_observed is False
    guard.assert_parameter_selection_allowed("construct StrategyAgent")  # must not raise
