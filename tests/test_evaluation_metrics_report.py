"""metrics_report.py's own job is the data-shape adapter from prediction_log
rows to the PRE-EXISTING metrics.py/portfolio_metrics.py dataclasses - these
tests build a small synthetic, already-resolved log directly via
db.save_prediction/db.record_prediction_outcome (bypassing outcome_resolution.py's own
timing gate, which has its own dedicated tests) so the adapter and
breakdown logic can be checked without waiting on real elapsed time."""
from datetime import datetime, timedelta, timezone

from market_agent.research.evaluation.metrics_report import (
    breakdown_by_contradiction, breakdown_by_dominant_axis, breakdown_by_magnitude, mode_report,
)
from market_agent.research.evaluation.modes import MODE_A
from market_agent.store import db

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _log_and_resolve(conn, entity, day, predicted_impact, realized_return, mode=MODE_A, news_state=None,
                      confidence=0.6):
    triggered_at = BASE + timedelta(days=day)
    pid = db.save_prediction(conn, entity=entity, mode=mode, triggered_at=triggered_at, model_version="test_v1",
                              decision_label="FAVORABLE" if predicted_impact and predicted_impact > 0 else "CAUTIOUS",
                              predicted_impact=predicted_impact, predicted_confidence=confidence,
                              inputs_snapshot={"entity": entity, "news_state": news_state})
    db.record_prediction_outcome(conn, pid, "realized_return_1d", "resolved_1d_at", realized_return, triggered_at)
    return pid


def test_mode_report_reflects_only_resolved_rows_for_that_horizon():
    conn = db.connect(":memory:")
    _log_and_resolve(conn, "ACME", 1, 1.0, 0.03)
    _log_and_resolve(conn, "ACME", 2, -1.0, -0.02)
    # log a third prediction but never resolve it - should be excluded from the 1d report entirely
    db.save_prediction(conn, entity="ACME", mode=MODE_A, triggered_at=BASE + timedelta(days=3),
                        model_version="test_v1", decision_label="FAVORABLE", predicted_impact=1.0,
                        predicted_confidence=0.6, inputs_snapshot={"entity": "ACME"})
    metrics, portfolio = mode_report(conn, MODE_A, 1)
    assert metrics.n == 2
    assert portfolio.n_trades == 2


def test_mode_report_is_honest_about_zero_resolved_observations():
    conn = db.connect(":memory:")
    db.save_prediction(conn, entity="ACME", mode=MODE_A, triggered_at=BASE, model_version="test_v1",
                        decision_label="FAVORABLE", predicted_impact=1.0, predicted_confidence=0.6,
                        inputs_snapshot={"entity": "ACME"})
    metrics, portfolio = mode_report(conn, MODE_A, 60)  # nothing resolved at this horizon yet
    assert metrics.n == 0
    assert portfolio.n_trades == 0


def test_breakdown_by_contradiction_separates_correctly():
    conn = db.connect(":memory:")
    _log_and_resolve(conn, "ACME", 1, 1.0, 0.03, news_state={"contradiction_axes": ["demand"]})
    _log_and_resolve(conn, "ACME", 2, 1.0, 0.02, news_state={"contradiction_axes": []})
    result = breakdown_by_contradiction(conn, MODE_A, 1)
    assert result["CONTRADICTION_PRESENT"].n == 1
    assert result["NO_CONTRADICTION"].n == 1


def test_breakdown_by_dominant_axis_groups_by_largest_magnitude_axis():
    conn = db.connect(":memory:")
    _log_and_resolve(conn, "ACME", 1, 0.5, 0.01,
                      news_state={"dimensions": {"growth": 0.9, "demand": 0.1}, "contradiction_axes": []})
    _log_and_resolve(conn, "ACME", 2, 0.5, 0.01,
                      news_state={"dimensions": {"risk": -0.8}, "contradiction_axes": []})
    result = breakdown_by_dominant_axis(conn, MODE_A, 1)
    assert result["growth"].n == 1
    assert result["risk"].n == 1


def test_breakdown_by_dominant_axis_excludes_rows_with_no_news_signal():
    conn = db.connect(":memory:")
    _log_and_resolve(conn, "ACME", 1, 1.0, 0.02, news_state=None)
    result = breakdown_by_dominant_axis(conn, MODE_A, 1)
    assert result == {}  # nothing to attribute - not fabricated into a fake bucket


def test_breakdown_by_magnitude_buckets_correctly():
    conn = db.connect(":memory:")
    _log_and_resolve(conn, "ACME", 1, 0.2, 0.01)    # SMALL
    _log_and_resolve(conn, "ACME", 2, 0.5, 0.01)    # MEDIUM
    _log_and_resolve(conn, "ACME", 3, 0.9, 0.01)    # LARGE
    result = breakdown_by_magnitude(conn, MODE_A, 1)
    assert result["SMALL"].n == 1
    assert result["MEDIUM"].n == 1
    assert result["LARGE"].n == 1
