from datetime import datetime, timezone

import pytest

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.governance import apply_test_results, promote_from_shadow, revalidate
from market_agent.learn.hypothesis_testing import HypothesisTestResult
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _hypothesis(conn, condition=None, concept=None, methodology_ids=None):
    event = EventRecord(entity="NVDA", event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=NOW,
                         ingested_at=NOW, context={"regime": "RISK_OFF"})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
    event_id = db.log_prediction(conn, event, pred)
    return db.add_hypothesis(conn, event_id, condition or {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                             "regime": "RISK_OFF"}, 20, "test", NOW,
                              concept=concept, methodology_ids=methodology_ids)


def test_confirmed_technical_hypothesis_propagates_concept_and_methodology_ids_to_the_relationship():
    """Regression test - apply_test_results() previously dropped the
    hypothesis's concept/methodology_ids entirely when promoting to
    SHADOW, which would have made every confirmed technical-concept or
    methodology-backed relationship invisible to agents/variants.py's
    ANY_TECHNICAL_CONCEPT/METHODOLOGY_BACKED_CONCEPT filters."""
    conn = db.connect(":memory:")
    hid = _hypothesis(conn, condition={"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                        "breakout_state": "BREAKOUT_DOWN"},
                       concept="BREAKOUT", methodology_ids=["meth-1", "meth-2"])
    result = HypothesisTestResult(hid, "CONFIRMED", n=30, mean_effect=-0.09, baseline_effect=-0.02,
                                   p_value=0.001, p_value_corrected=0.004, ci_low=-0.11, ci_high=-0.07,
                                   evidence=["strong effect"])
    promoted = apply_test_results(conn, [result], promoted_by="test-suite", clock_now=NOW)
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?", (promoted[0],)).fetchone()
    assert rel["concept"] == "BREAKOUT"
    import json
    assert json.loads(rel["methodology_ids_json"]) == ["meth-1", "meth-2"]


def test_confirmed_result_enters_shadow_not_active_immediately():
    """Shadow deployment (Blueprint §21): a confirmed hypothesis must
    prove itself on new, disjoint evidence before going live - it does
    NOT skip straight to ACTIVE. See learn/shadow.py."""
    conn = db.connect(":memory:")
    hid = _hypothesis(conn)
    result = HypothesisTestResult(hid, "CONFIRMED", n=30, mean_effect=-0.09, baseline_effect=-0.02,
                                   p_value=0.001, p_value_corrected=0.004, ci_low=-0.11, ci_high=-0.07,
                                   evidence=["strong effect"])
    promoted = apply_test_results(conn, [result], promoted_by="test-suite", clock_now=NOW)
    assert len(promoted) == 1
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?", (promoted[0],)).fetchone()
    assert rel["status"] == "SHADOW"
    assert rel["effect_estimate"] == -0.09
    assert rel["ci_low"] == -0.11 and rel["ci_high"] == -0.07
    assert rel["shadow_started_at"] is not None
    registry = conn.execute("SELECT * FROM model_registry").fetchall()
    assert len(registry) == 1
    assert registry[0]["promotion_status"] == "SHADOW"

    hyp_row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()
    assert hyp_row["status"] == "CONFIRMED"


def test_promote_from_shadow_to_active_on_passing_new_evidence():
    conn = db.connect(":memory:")
    hid = _hypothesis(conn)
    shadow_result = HypothesisTestResult(hid, "CONFIRMED", n=30, mean_effect=-0.09, baseline_effect=-0.02,
                                          p_value=0.001, p_value_corrected=0.004, evidence=["strong effect"])
    [relationship_id] = apply_test_results(conn, [shadow_result], promoted_by="test-suite", clock_now=NOW)

    new_evidence_result = HypothesisTestResult(relationship_id, "CONFIRMED", n=12, mean_effect=-0.085,
                                                baseline_effect=-0.02, p_value=0.01, p_value_corrected=0.01,
                                                ci_low=-0.10, ci_high=-0.07, evidence=["new evidence still holds"])
    new_status = promote_from_shadow(conn, relationship_id, new_evidence_result, promoted_by="test-suite",
                                      clock_now=NOW)
    assert new_status == "ACTIVE"
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?", (relationship_id,)).fetchone()
    assert rel["status"] == "ACTIVE"
    assert rel["shadow_promoted_at"] is not None
    assert rel["n_supporting"] == 12  # updated to the new-evidence-only N, not the original shadow-entry N


def test_promote_from_shadow_retires_directly_on_failing_new_evidence():
    """A relationship that fails its shadow probation is retired having
    NEVER influenced a live prediction - distinct from an ACTIVE
    relationship later retired by ordinary revalidation."""
    conn = db.connect(":memory:")
    hid = _hypothesis(conn)
    shadow_result = HypothesisTestResult(hid, "CONFIRMED", n=30, mean_effect=-0.09, baseline_effect=-0.02,
                                          p_value=0.001, p_value_corrected=0.004, evidence=["strong effect"])
    [relationship_id] = apply_test_results(conn, [shadow_result], promoted_by="test-suite", clock_now=NOW)

    failing_result = HypothesisTestResult(relationship_id, "REJECTED_NOT_SIGNIFICANT", n=12, mean_effect=-0.022,
                                           baseline_effect=-0.02, p_value=0.8, p_value_corrected=0.8,
                                           evidence=["did not replicate"])
    new_status = promote_from_shadow(conn, relationship_id, failing_result, promoted_by="test-suite", clock_now=NOW)
    assert new_status == "RETIRED"
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?", (relationship_id,)).fetchone()
    assert rel["status"] == "RETIRED"
    assert rel["shadow_promoted_at"] is None  # never went live


def test_promote_from_shadow_rejects_a_non_shadow_relationship():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 30, "ACTIVE", NOW)
    result = HypothesisTestResult("rel-1", "CONFIRMED", n=12, mean_effect=-0.085, baseline_effect=-0.02,
                                   p_value=0.01, p_value_corrected=0.01, evidence=[])
    with pytest.raises(ValueError):
        promote_from_shadow(conn, "rel-1", result, promoted_by="test-suite", clock_now=NOW)


def test_rejected_result_never_creates_a_relationship_or_registry_entry():
    conn = db.connect(":memory:")
    hid = _hypothesis(conn)
    result = HypothesisTestResult(hid, "REJECTED_NOT_SIGNIFICANT", n=20, mean_effect=-0.021,
                                   baseline_effect=-0.02, p_value=0.6, p_value_corrected=0.9, evidence=["noise"])
    promoted = apply_test_results(conn, [result], promoted_by="test-suite", clock_now=NOW)
    assert promoted == []
    assert conn.execute("SELECT COUNT(*) c FROM validated_relationships").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM model_registry").fetchone()["c"] == 0

    hyp_row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()
    assert hyp_row["status"] == "REJECTED"
    assert hyp_row["test_result_json"] is not None  # the rejection reason is permanently recorded


def test_revalidate_confirming_result_keeps_relationship_active_and_updates_stats():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 30, "ACTIVE", NOW)
    new_result = HypothesisTestResult("rel-1", "CONFIRMED", n=60, mean_effect=-0.085, baseline_effect=-0.02,
                                       p_value=0.0001, p_value_corrected=0.0003, evidence=["still holds"])
    status = revalidate(conn, "rel-1", new_result, promoted_by="test-suite", clock_now=NOW)
    assert status == "ACTIVE"
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    assert rel["n_supporting"] == 60
    assert rel["last_revalidated_at"] is not None


def test_revalidate_failing_result_retires_relationship_never_deletes_it():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 30, "ACTIVE", NOW)
    new_result = HypothesisTestResult("rel-1", "REJECTED_NOT_SIGNIFICANT", n=60, mean_effect=-0.025,
                                       baseline_effect=-0.02, p_value=0.4, p_value_corrected=0.7,
                                       evidence=["stopped replicating"])
    status = revalidate(conn, "rel-1", new_result, promoted_by="test-suite", clock_now=NOW)
    assert status == "RETIRED"
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    assert rel["status"] == "RETIRED"
    assert rel is not None  # row still exists - retirement is not deletion

    registry = conn.execute("SELECT * FROM model_registry WHERE change_json LIKE '%REVALIDATE%'").fetchall()
    assert len(registry) == 1
    assert registry[0]["promotion_status"] == "ROLLED_BACK"
