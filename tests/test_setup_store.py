from datetime import datetime, timezone

import pytest

from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    return db.connect(":memory:")


def test_log_and_query_setup_observation(conn):
    obs_id = db.log_setup_observation(conn, "ACME", NOW, "RISK_ON", {"trend_direction": "UP"}, 20)
    row = db.get_setup_observation(conn, obs_id)
    assert row["entity"] == "ACME"
    assert row["regime"] == "RISK_ON"
    assert row["outcome_locked"] == 0
    assert row["realized_abnormal_return"] is None


def test_record_setup_outcome_locks_and_is_append_only(conn):
    obs_id = db.log_setup_observation(conn, "ACME", NOW, "RISK_ON", {"trend_direction": "UP"}, 20)
    db.record_setup_outcome(conn, obs_id, 0.03, NOW)
    row = db.get_setup_observation(conn, obs_id)
    assert row["outcome_locked"] == 1
    assert row["realized_abnormal_return"] == 0.03

    with pytest.raises(db.AppendOnlyViolation):
        db.record_setup_outcome(conn, obs_id, 0.05, NOW)


def test_record_setup_outcome_missing_id_raises(conn):
    with pytest.raises(KeyError):
        db.record_setup_outcome(conn, "not-a-real-id", 0.01, NOW)


def test_query_setup_observations_filters(conn):
    db.log_setup_observation(conn, "A", NOW, "RISK_ON", {}, 20)
    obs2 = db.log_setup_observation(conn, "B", NOW, "RISK_ON", {}, 60)
    db.record_setup_outcome(conn, obs2, 0.02, NOW)

    all_rows = db.query_setup_observations(conn)
    assert len(all_rows) == 2

    horizon_20 = db.query_setup_observations(conn, horizon_days=20)
    assert len(horizon_20) == 1

    resolved_only = db.query_setup_observations(conn, outcome_known_only=True)
    assert len(resolved_only) == 1
    assert resolved_only[0]["entity"] == "B"


def test_upsert_and_get_discovered_setup(conn):
    db.upsert_discovered_setup(conn, "S_001", "RISK_ON", {"trend_direction": "UP"}, 20, 0.05,
                                train_result={"status": "SCREENED"}, validate_result=None, shadow_result=None,
                                test_result=None, status="TRAIN_SCREENED", created_at=NOW)
    row = db.get_discovered_setup(conn, "S_001")
    assert row["status"] == "TRAIN_SCREENED"
    assert row["regime"] == "RISK_ON"

    # upsert again with more results, escalating status - existing fields preserved/overwritten as expected
    db.upsert_discovered_setup(conn, "S_001", "RISK_ON", {"trend_direction": "UP"}, 20, 0.05,
                                train_result={"status": "SCREENED"}, validate_result={"status": "CONFIRMED"},
                                shadow_result=None, test_result=None, status="VALIDATED", created_at=NOW)
    row = db.get_discovered_setup(conn, "S_001")
    assert row["status"] == "VALIDATED"
    import json
    assert json.loads(row["validate_result_json"])["status"] == "CONFIRMED"


def test_discovered_setups_by_status(conn):
    db.upsert_discovered_setup(conn, "S_A", None, {}, 20, None, None, None, None, None, "TEST_VALIDATED", NOW)
    db.upsert_discovered_setup(conn, "S_B", None, {}, 20, None, None, None, None, None, "REJECTED", NOW)
    validated = db.discovered_setups_by_status(conn, "TEST_VALIDATED")
    assert {r["setup_id"] for r in validated} == {"S_A"}
