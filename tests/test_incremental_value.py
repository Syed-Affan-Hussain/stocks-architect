from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.incremental_value import test_incremental_value as run_incremental_value_test
from market_agent.learn.hypothesis_testing import MIN_N
from market_agent.store import db

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
CONDITION = {"event_type": "GUIDANCE_CHANGE", "direction": "positive", "breakout_state": "BREAKOUT_UP"}


def _log(conn, entity, model_version, published, predicted_impact, realized):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                         source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                         ingested_at=published, context={"regime": "RISK_ON", "breakout_state": "BREAKOUT_UP"})
    pred = PredictionRecord(20, predicted_impact, "MEDIUM", {"basis": "unconditional_baseline"},
                             model_version, published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized, published + timedelta(days=20), realized - predicted_impact, "OK")
    return event_id


def _seed_matching_rows(conn, n, current_adaptive_pred_fn, realized_fn):
    rows = []
    for i in range(n):
        published = NOW + timedelta(days=3 * i)
        _log(conn, f"E{i}", "CURRENT_ADAPTIVE_v1", published, current_adaptive_pred_fn(i), realized_fn(i))
        # a second agent's row for the SAME real event - the technical-concept relationship's
        # matching row (deduplication means only one of the two survives, but test_incremental_value
        # always looks up CURRENT_ADAPTIVE's own row independently, so it doesn't matter which)
        eid = _log(conn, f"E{i}", "TECHNICAL_ADAPTIVE_v1", published, current_adaptive_pred_fn(i), realized_fn(i))
        rows.append(db.get_event(conn, eid))
    return rows


def test_insufficient_n_when_few_matching_cases():
    conn = db.connect(":memory:")
    rows = _seed_matching_rows(conn, 5, lambda i: 0.02, lambda i: 0.02)
    result = run_incremental_value_test(conn, CONDITION, 20, rows)
    assert result.status == "INSUFFICIENT_N"
    assert result.n == 5


def test_no_incremental_value_when_existing_model_already_tracks_realized_returns():
    """CURRENT_ADAPTIVE's own prediction already matches the realized
    return closely - the technical condition adds nothing beyond it."""
    conn = db.connect(":memory:")
    rows = _seed_matching_rows(conn, 20, lambda i: 0.05 + (0.001 if i % 2 == 0 else -0.001),
                                lambda i: 0.05 + (0.001 if i % 2 == 0 else -0.001))
    result = run_incremental_value_test(conn, CONDITION, 20, rows)
    assert result.status == "NO_INCREMENTAL_VALUE"
    assert result.n == 20


def test_incremental_value_confirmed_when_existing_model_systematically_undershoots():
    """CURRENT_ADAPTIVE consistently predicts +2% but reality is +10% for
    these specific cases - the technical condition captures real
    information the existing model is missing."""
    conn = db.connect(":memory:")
    rows = _seed_matching_rows(conn, 20, lambda i: 0.02,
                                lambda i: 0.10 + (0.005 if i % 2 == 0 else -0.005))
    result = run_incremental_value_test(conn, CONDITION, 20, rows)
    assert result.status == "INCREMENTAL_VALUE_CONFIRMED"
    assert result.mean_incremental_diff > 0.05


def test_rows_without_a_current_adaptive_prediction_are_skipped_not_crashed():
    conn = db.connect(":memory:")
    # log TECHNICAL_ADAPTIVE rows with NO corresponding CURRENT_ADAPTIVE row at all
    rows = []
    for i in range(5):
        published = NOW + timedelta(days=3 * i)
        eid = _log(conn, f"E{i}", "TECHNICAL_ADAPTIVE_v1", published, 0.05, 0.10)
        rows.append(db.get_event(conn, eid))
    result = run_incremental_value_test(conn, CONDITION, 20, rows)
    assert result.status == "INSUFFICIENT_N"
    assert result.n == 0


def test_evidence_never_asserts_promotion_only_reports_diagnostic():
    conn = db.connect(":memory:")
    rows = _seed_matching_rows(conn, 20, lambda i: 0.02, lambda i: 0.10)
    result = run_incremental_value_test(conn, CONDITION, 20, rows)
    assert any("not itself a promotion gate" in e for e in result.evidence)
