from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.experiment.rolling_monitor import (
    STANDARD_WINDOW_SIZES, rolling_by_dimension, rolling_comparison,
)
from market_agent.store import db

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _log(conn, model_version, event_type, direction, regime, horizon_days, predicted_impact,
         realized, i):
    published = START + timedelta(days=i)
    event = EventRecord(entity=f"E{i}", event_type=event_type, direction=direction, source="wire",
                         source_reliability_snapshot=0.5, raw_text="x", published_at=published,
                         ingested_at=published, context={"regime": regime})
    pred = PredictionRecord(horizon_days, predicted_impact, "MEDIUM", {"basis": "unconditional_baseline"},
                             model_version, published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized, published + timedelta(days=horizon_days),
                       realized - predicted_impact, "OK")
    return event_id


def _seed(conn, n=12):
    for i in range(n):
        _log(conn, "STATIC_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, i)
        _log(conn, "ADAPTIVE_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.03, -0.03, i + 1000)


def test_rolling_comparison_uses_only_resolved_predictions_per_agent():
    conn = db.connect(":memory:")
    _seed(conn, n=12)
    comparisons = rolling_comparison(conn, window_sizes=(5, 50))
    assert [c.window_size for c in comparisons] == [5, 50]
    five = comparisons[0]
    assert five.n_available == 5
    assert five.static_metrics.n == 5
    assert five.adaptive_metrics.n == 5
    fifty = comparisons[1]
    assert fifty.n_available == 12  # fewer resolved predictions exist than the window size
    assert fifty.is_descriptive_only is True


def test_rolling_comparison_default_windows_are_the_standard_sizes():
    conn = db.connect(":memory:")
    _seed(conn, n=3)
    comparisons = rolling_comparison(conn)
    assert [c.window_size for c in comparisons] == list(STANDARD_WINDOW_SIZES)
    assert all(c.n_available == 3 for c in comparisons)


def test_rolling_comparison_only_counts_matching_model_version_prefix():
    conn = db.connect(":memory:")
    _log(conn, "STATIC_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 0)
    _log(conn, "STATIC_v2_experimental", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 1)
    _log(conn, "ADAPTIVE_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 2)
    comparisons = rolling_comparison(conn, window_sizes=(10,))
    assert comparisons[0].static_metrics.n == 2  # both STATIC_v1 and STATIC_v2_experimental
    assert comparisons[0].adaptive_metrics.n == 1


def test_rolling_by_dimension_splits_by_event_type():
    conn = db.connect(":memory:")
    _log(conn, "STATIC_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 0)
    _log(conn, "ADAPTIVE_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 1)
    _log(conn, "STATIC_v1", "EARNINGS_RESULT", "positive", "RISK_ON", 5, 0.01, 0.02, 2)
    _log(conn, "ADAPTIVE_v1", "EARNINGS_RESULT", "positive", "RISK_ON", 5, 0.01, 0.02, 3)
    reports = rolling_by_dimension(conn, "event_type", window_sizes=(10,))
    values = {r.dimension_value for r in reports}
    assert values == {"GUIDANCE_CHANGE", "EARNINGS_RESULT"}
    for r in reports:
        assert r.windows[0].static_metrics.n == 1
        assert r.windows[0].adaptive_metrics.n == 1


def test_rolling_by_dimension_regime_reads_from_context_json():
    conn = db.connect(":memory:")
    _log(conn, "STATIC_v1", "GUIDANCE_CHANGE", "negative", "RISK_OFF", 20, -0.02, -0.03, 0)
    _log(conn, "STATIC_v1", "GUIDANCE_CHANGE", "negative", "RISK_ON", 20, -0.02, -0.03, 1)
    reports = rolling_by_dimension(conn, "regime", window_sizes=(10,))
    assert {r.dimension_value for r in reports} == {"RISK_OFF", "RISK_ON"}


def test_rolling_by_dimension_rejects_unknown_dimension():
    conn = db.connect(":memory:")
    try:
        rolling_by_dimension(conn, "not_a_real_dimension")
        assert False, "expected ValueError"
    except ValueError:
        pass
