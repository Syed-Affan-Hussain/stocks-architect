from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.revalidation import run_revalidation_pass
from market_agent.store import db

BASELINE = {20: 0.02}
START = datetime(2023, 1, 1, tzinfo=timezone.utc)
CHECK_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _log_resolved(conn, entity, published, realized, regime="RISK_OFF"):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                         ingested_at=published, context={"regime": regime})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized, published + timedelta(days=20), realized + 0.02, "OK")


def test_revalidation_keeps_a_still_replicating_relationship_active():
    conn = db.connect(":memory:")
    for i in range(20):
        _log_resolved(conn, f"PEER{i}", START + timedelta(days=15 * i), -0.09 + (0.001 if i % 2 == 0 else -0.001))
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.08, None, None, 15, "ACTIVE", START)

    summary = run_revalidation_pass(conn, BASELINE, promoted_by="test-suite", clock_now=CHECK_DATE)
    assert len(summary) == 1
    assert summary[0]["new_status"] == "ACTIVE"
    assert summary[0]["n_after"] == 20  # picked up all the matching history, not just what it was promoted with

    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    assert rel["status"] == "ACTIVE"
    assert rel["last_revalidated_at"] is not None


def test_revalidation_retires_a_relationship_that_stopped_replicating():
    conn = db.connect(":memory:")
    for i in range(20):
        _log_resolved(conn, f"PEER{i}", START + timedelta(days=15 * i),
                      -0.021 + (0.0005 if i % 2 == 0 else -0.0005))  # now indistinguishable from baseline
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.08, None, None, 15, "ACTIVE", START)

    summary = run_revalidation_pass(conn, BASELINE, promoted_by="test-suite", clock_now=CHECK_DATE)
    assert summary[0]["new_status"] == "RETIRED"

    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    assert rel["status"] == "RETIRED"
    assert rel is not None  # retirement, never deletion


def test_revalidation_ignores_already_retired_relationships():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.08, None, None, 15, "RETIRED", START)
    summary = run_revalidation_pass(conn, BASELINE, promoted_by="test-suite", clock_now=CHECK_DATE)
    assert summary == []


def test_revalidation_never_touches_episodic_events():
    conn = db.connect(":memory:")
    for i in range(20):
        _log_resolved(conn, f"PEER{i}", START + timedelta(days=15 * i), -0.09 + (0.001 if i % 2 == 0 else -0.001))
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.08, None, None, 15, "ACTIVE", START)

    before = conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"]
    run_revalidation_pass(conn, BASELINE, promoted_by="test-suite", clock_now=CHECK_DATE)
    after = conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"]
    assert before == after == 20
