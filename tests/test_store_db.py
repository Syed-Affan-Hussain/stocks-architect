from datetime import datetime, timezone

import pytest

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.store import db


def _event(entity="NVDA", direction="negative", published=datetime(2024, 1, 10, tzinfo=timezone.utc)):
    return EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction=direction, source="test-source",
                        source_reliability_snapshot=0.5, raw_text="Company cuts guidance",
                        published_at=published, ingested_at=published,
                        context={"regime": "NORMAL", "prior_5d_return": -0.01, "sector_momentum": "NEUTRAL"})


def _prediction(horizon_days=20, impact=-0.02):
    return PredictionRecord(horizon_days=horizon_days, predicted_impact=impact, predicted_confidence="MEDIUM",
                             basis={"basis": "unconditional_baseline"}, model_version="TEST_v1",
                             predicted_at=datetime(2024, 1, 10, tzinfo=timezone.utc))


@pytest.fixture
def conn():
    return db.connect(":memory:")


def test_log_prediction_and_get_event_roundtrip(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    row = db.get_event(conn, event_id)
    assert row["entity"] == "NVDA"
    assert row["event_type"] == "GUIDANCE_CHANGE"
    assert row["predicted_impact"] == -0.02
    assert row["outcome_locked"] == 0
    assert row["realized_abnormal_return"] is None


def test_record_outcome_sets_fields_and_locks_row(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    db.record_outcome(conn, event_id, realized_abnormal_return=-0.05, observed_at=datetime(2024, 1, 30, tzinfo=timezone.utc),
                       error_value=-0.03, error_type="WRONG_MAGNITUDE")
    row = db.get_event(conn, event_id)
    assert row["realized_abnormal_return"] == -0.05
    assert row["error_type"] == "WRONG_MAGNITUDE"
    assert row["outcome_locked"] == 1


def test_record_outcome_twice_raises_append_only_violation(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    db.record_outcome(conn, event_id, -0.05, datetime(2024, 1, 30, tzinfo=timezone.utc), -0.03, "WRONG_MAGNITUDE")
    with pytest.raises(db.AppendOnlyViolation):
        db.record_outcome(conn, event_id, 0.10, datetime(2024, 1, 31, tzinfo=timezone.utc), 0.13, "WRONG_DIRECTION")
    # the FIRST outcome must survive untouched
    row = db.get_event(conn, event_id)
    assert row["realized_abnormal_return"] == -0.05


def test_record_outcome_unknown_event_id_raises_key_error(conn):
    with pytest.raises(KeyError):
        db.record_outcome(conn, "does-not-exist", 0.0, datetime(2024, 1, 1, tzinfo=timezone.utc), 0.0, "OK")


def test_query_events_filters_by_published_before(conn):
    db.log_prediction(conn, _event(published=datetime(2024, 1, 1, tzinfo=timezone.utc)), _prediction())
    db.log_prediction(conn, _event(published=datetime(2024, 6, 1, tzinfo=timezone.utc)), _prediction())
    rows = db.query_events(conn, published_before=datetime(2024, 3, 1, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert rows[0]["published_at"].startswith("2024-01-01")


def test_query_events_outcome_known_only(conn):
    e1 = db.log_prediction(conn, _event(), _prediction())
    db.log_prediction(conn, _event(entity="AAPL"), _prediction())
    db.record_outcome(conn, e1, -0.05, datetime(2024, 1, 30, tzinfo=timezone.utc), -0.03, "WRONG_MAGNITUDE")
    rows = db.query_events(conn, outcome_known_only=True)
    assert len(rows) == 1
    assert rows[0]["entity"] == "NVDA"


def test_upsert_relationship_and_active_relationships(conn):
    db.upsert_relationship(conn, "rel-1", condition={"event_type": "GUIDANCE_CHANGE", "direction": "negative"},
                            horizon_days=20, effect_estimate=-0.04, ci_low=-0.06, ci_high=-0.02, n_supporting=30,
                            status="ACTIVE", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    active = db.active_relationships(conn, "GUIDANCE_CHANGE", 20)
    assert len(active) == 1
    assert active[0]["relationship_id"] == "rel-1"


def test_active_relationships_excludes_retired(conn):
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                            -0.04, None, None, 30, "RETIRED", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert db.active_relationships(conn, "GUIDANCE_CHANGE", 20) == []


def test_hypothesis_lifecycle(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    hid = db.add_hypothesis(conn, source_event_id=event_id, condition={"event_type": "GUIDANCE_CHANGE"},
                             horizon_days=20, explanation_text="test", proposed_at=datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert len(db.untested_hypotheses(conn)) == 1
    db.set_hypothesis_result(conn, hid, "CONFIRMED", datetime(2024, 3, 1, tzinfo=timezone.utc), {"p": 0.01})
    assert db.untested_hypotheses(conn) == []


def test_set_hypothesis_result_rejects_invalid_status(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    hid = db.add_hypothesis(conn, event_id, {"event_type": "GUIDANCE_CHANGE"}, 20, "test",
                             datetime(2024, 2, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        db.set_hypothesis_result(conn, hid, "MAYBE", datetime(2024, 3, 1, tzinfo=timezone.utc), {})


def test_register_change_records_governance_entry(conn):
    db.register_change(conn, "v1", reason="test promotion", change={"action": "CREATE_RELATIONSHIP"},
                        performance_before=None, performance_after={"n": 20}, statistical_tests={"p": 0.01},
                        promoted_by="test-suite", promotion_status="PROMOTED",
                        created_at=datetime(2024, 3, 1, tzinfo=timezone.utc))
    row = conn.execute("SELECT * FROM model_registry WHERE version_id = 'v1'").fetchone()
    assert row["promotion_status"] == "PROMOTED"
    assert row["promoted_by"] == "test-suite"


# --- stage 6: methodology/concept provenance ---

def test_add_methodology_and_get_roundtrip(conn):
    db.add_methodology(conn, "meth-1", name="Test System", practitioner="Test Trader",
                        source_type="book", source_description="A paraphrased summary.",
                        extractor_name="RULE_BASED", ingested_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    row = db.get_methodology(conn, "meth-1")
    assert row["name"] == "Test System"
    assert row["practitioner"] == "Test Trader"
    assert row["extractor_name"] == "RULE_BASED"


def test_add_methodology_is_idempotent_on_conflict(conn):
    db.add_methodology(conn, "meth-1", "Test System", "Test Trader", "book", "desc", "RULE_BASED",
                        datetime(2024, 1, 1, tzinfo=timezone.utc))
    db.add_methodology(conn, "meth-1", "Test System", "Test Trader", "book", "desc", "RULE_BASED",
                        datetime(2024, 1, 1, tzinfo=timezone.utc))  # second call must not raise
    assert len(db.all_methodologies(conn)) == 1


def test_methodology_concept_links_and_reverse_lookup(conn):
    db.add_methodology(conn, "meth-1", "System A", "Trader A", "book", "desc", "RULE_BASED",
                        datetime(2024, 1, 1, tzinfo=timezone.utc))
    db.add_methodology(conn, "meth-2", "System B", "Trader B", "interview", "desc", "RULE_BASED",
                        datetime(2024, 1, 2, tzinfo=timezone.utc))
    db.add_methodology_concept_link(conn, "link-1", "meth-1", "BREAKOUT", "mentions breakouts",
                                     datetime(2024, 1, 1, tzinfo=timezone.utc))
    db.add_methodology_concept_link(conn, "link-2", "meth-2", "BREAKOUT", "also mentions breakouts",
                                     datetime(2024, 1, 2, tzinfo=timezone.utc))
    db.add_methodology_concept_link(conn, "link-3", "meth-1", "TREND", "mentions trend",
                                     datetime(2024, 1, 1, tzinfo=timezone.utc))

    breakout_methodologies = db.methodologies_for_concept(conn, "BREAKOUT")
    assert {r["methodology_id"] for r in breakout_methodologies} == {"meth-1", "meth-2"}

    links = db.concept_links_for_methodology(conn, "meth-1")
    assert {r["concept"] for r in links} == {"BREAKOUT", "TREND"}


def test_add_hypothesis_stores_concept_and_methodology_provenance(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    hid = db.add_hypothesis(conn, event_id, {"event_type": "GUIDANCE_CHANGE", "breakout_state": "BREAKOUT_UP"},
                             20, "test", datetime(2024, 2, 1, tzinfo=timezone.utc),
                             concept="BREAKOUT", methodology_ids=["meth-1", "meth-2"])
    row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()
    assert row["concept"] == "BREAKOUT"
    import json
    assert json.loads(row["methodology_ids_json"]) == ["meth-1", "meth-2"]


def test_add_hypothesis_without_concept_leaves_it_null(conn):
    event_id = db.log_prediction(conn, _event(), _prediction())
    hid = db.add_hypothesis(conn, event_id, {"event_type": "GUIDANCE_CHANGE"}, 20, "test",
                             datetime(2024, 2, 1, tzinfo=timezone.utc))
    row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()
    assert row["concept"] is None
    assert row["methodology_ids_json"] is None


def test_upsert_relationship_stores_concept_and_methodology_provenance(conn):
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "breakout_state": "BREAKOUT_UP"},
                            20, 0.05, None, None, 20, "SHADOW", datetime(2024, 1, 1, tzinfo=timezone.utc),
                            concept="BREAKOUT", methodology_ids=["meth-1"])
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = 'rel-1'").fetchone()
    assert row["concept"] == "BREAKOUT"
    import json
    assert json.loads(row["methodology_ids_json"]) == ["meth-1"]
