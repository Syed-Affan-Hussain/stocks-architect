"""Regression coverage for store/db.py::deduplicate_by_real_event -
found by direct inspection of a real four-agent walk-forward run, where
one condition's "matching rows" for a hypothesis test were exactly 4x the
number of distinct real events (each of STATIC/CURRENT_ADAPTIVE/
TECHNICAL_ADAPTIVE/METHODOLOGY_ADAPTIVE logs its own episodic_events row
per real event). This bug predates stage 6-7 - even the original 2-agent
(STATIC/ADAPTIVE) walk-forward inflated N by 2x - and directly affects
every module that queries episodic_events to build a statistical test's
sample: learn/hypothesis_testing.py, learn/shadow.py,
learn/hierarchical_research.py, retrieval/similarity.py,
reporting/knowledge_state.py.
"""
from datetime import datetime, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.store import db

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _log(conn, entity, model_version, published):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                         ingested_at=published, context={"regime": "RISK_OFF"})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, model_version, published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, -0.03, published, -0.01, "OK")
    return event_id


def test_deduplicates_multiple_agent_rows_for_the_same_real_event():
    conn = db.connect(":memory:")
    published = NOW
    for model_version in ("STATIC_v1", "CURRENT_ADAPTIVE_v1", "TECHNICAL_ADAPTIVE_v1", "METHODOLOGY_ADAPTIVE_v1"):
        _log(conn, "ACME", model_version, published)
    rows = conn.execute("SELECT * FROM episodic_events").fetchall()
    assert len(rows) == 4
    deduped = db.deduplicate_by_real_event(rows)
    assert len(deduped) == 1


def test_keeps_genuinely_distinct_real_events():
    conn = db.connect(":memory:")
    from datetime import timedelta
    for i in range(5):
        published = NOW + timedelta(days=i)
        for model_version in ("STATIC_v1", "ADAPTIVE_v1"):
            _log(conn, f"E{i}", model_version, published)
    rows = conn.execute("SELECT * FROM episodic_events").fetchall()
    assert len(rows) == 10
    deduped = db.deduplicate_by_real_event(rows)
    assert len(deduped) == 5


def test_single_prediction_per_event_is_a_no_op():
    """Isolated unit-test fixtures that only ever log one prediction per
    real event (the common pattern across this project's other test
    files) must be completely unaffected."""
    conn = db.connect(":memory:")
    from datetime import timedelta
    for i in range(3):
        _log(conn, f"E{i}", "TEST_v1", NOW + timedelta(days=i))
    rows = conn.execute("SELECT * FROM episodic_events").fetchall()
    deduped = db.deduplicate_by_real_event(rows)
    assert len(deduped) == 3


def test_dedup_key_includes_horizon_so_different_horizons_never_collide():
    conn = db.connect(":memory:")
    event = EventRecord(entity="ACME", event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=NOW,
                         ingested_at=NOW, context={"regime": "RISK_OFF"})
    for horizon in (5, 20):
        pred = PredictionRecord(horizon, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
        event_id = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, event_id, -0.03, NOW, -0.01, "OK")
    rows = conn.execute("SELECT * FROM episodic_events").fetchall()
    deduped = db.deduplicate_by_real_event(rows)
    assert len(deduped) == 2  # two genuinely independent horizons, not deduplicated together


def test_deduplication_is_deterministic_not_arbitrary():
    """Picks the same row (smallest event_id) regardless of input order -
    never 'whichever agent's row happens to look best'."""
    conn = db.connect(":memory:")
    ids = [_log(conn, "ACME", v, NOW) for v in ("STATIC_v1", "CURRENT_ADAPTIVE_v1")]
    rows = conn.execute("SELECT * FROM episodic_events").fetchall()
    result_a = db.deduplicate_by_real_event(rows)
    result_b = db.deduplicate_by_real_event(list(reversed(rows)))
    assert result_a[0]["event_id"] == result_b[0]["event_id"] == min(ids)
