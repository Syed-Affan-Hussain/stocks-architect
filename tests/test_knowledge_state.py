from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.reporting.knowledge_state import build_knowledge_state_report
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_report_separates_active_shadow_and_retired_relationships():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-active", {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                            -0.09, -0.11, -0.07, 40, "ACTIVE", NOW, last_revalidated_at=NOW)
    db.upsert_relationship(conn, "rel-shadow", {"event_type": "GUIDANCE_CHANGE", "direction": "positive"}, 20,
                            0.05, None, None, 20, "SHADOW", NOW, shadow_started_at=NOW)
    db.upsert_relationship(conn, "rel-retired", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                  "regime": "RISK_ON"}, 20, -0.03, None, None, 18, "RETIRED", NOW)

    report = build_knowledge_state_report(conn, now=NOW)
    assert {r.relationship_id for r in report.active_relationships} == {"rel-active"}
    assert {r.relationship_id for r in report.shadow_relationships} == {"rel-shadow"}
    assert {r.relationship_id for r in report.retired_relationships} == {"rel-retired"}


def test_decay_state_classification():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "fresh", {"event_type": "X", "direction": "positive"}, 20, 0.05, None, None,
                            20, "ACTIVE", NOW - timedelta(days=200), last_revalidated_at=NOW - timedelta(days=10))
    db.upsert_relationship(conn, "due", {"event_type": "Y", "direction": "positive"}, 20, 0.05, None, None,
                            20, "ACTIVE", NOW - timedelta(days=200), last_revalidated_at=NOW - timedelta(days=150))
    db.upsert_relationship(conn, "overdue", {"event_type": "Z", "direction": "positive"}, 20, 0.05, None, None,
                            20, "ACTIVE", NOW - timedelta(days=400), last_revalidated_at=NOW - timedelta(days=250))
    db.upsert_relationship(conn, "never", {"event_type": "W", "direction": "positive"}, 20, 0.05, None, None,
                            20, "ACTIVE", NOW)

    report = build_knowledge_state_report(conn, now=NOW)
    by_id = {r.relationship_id: r.decay_state for r in report.active_relationships}
    assert by_id["fresh"] == "FRESH"
    assert by_id["due"] == "DUE_FOR_REVALIDATION"
    assert by_id["overdue"] == "OVERDUE"
    assert by_id["never"] == "NEVER_REVALIDATED"


def test_prediction_support_and_contradiction_counts():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                            -0.09, None, None, 40, "ACTIVE", NOW)

    def _log_and_resolve(entity, realized, error_type):
        event = EventRecord(entity, "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                             {"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.09, "HIGH", {"basis": "validated_relationship", "relationship_id": "rel-1"},
                                 "ADAPTIVE_v1", NOW)
        eid = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, eid, realized, NOW, realized - (-0.09), error_type)

    _log_and_resolve("A", -0.095, "OK")
    _log_and_resolve("B", -0.09, "OK")
    _log_and_resolve("C", 0.05, "WRONG_DIRECTION")  # this relationship led the prediction astray here

    report = build_knowledge_state_report(conn, now=NOW)
    rel = report.active_relationships[0]
    assert rel.n_predictions_supported == 3
    assert rel.n_predictions_contradicted == 1


def test_source_reliability_summary():
    conn = db.connect(":memory:")

    def _log_and_resolve(source, realized, error_type):
        event = EventRecord("A", "GUIDANCE_CHANGE", "negative", source, 0.5, "cuts guidance", NOW, NOW,
                             {"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
        eid = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, eid, realized, NOW, realized - (-0.02), error_type)

    _log_and_resolve("wire-A", -0.02, "OK")
    _log_and_resolve("wire-A", -0.03, "OK")
    _log_and_resolve("wire-A", 0.10, "WRONG_DIRECTION")
    _log_and_resolve("wire-B", -0.02, "OK")

    report = build_knowledge_state_report(conn, now=NOW)
    by_source = {s.source: s for s in report.source_reliability}
    assert by_source["wire-A"].n_resolved_predictions == 3
    assert by_source["wire-A"].n_learnable_errors == 1
    assert abs(by_source["wire-A"].hit_rate - (2 / 3)) < 1e-9
    assert by_source["wire-B"].hit_rate == 1.0


def test_rejected_hypotheses_are_listed():
    conn = db.connect(":memory:")
    event = EventRecord("A", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                         {"regime": "RISK_OFF"})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
    eid = db.log_prediction(conn, event, pred)
    hid = db.add_hypothesis(conn, eid, {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                             "test explanation", NOW)
    db.set_hypothesis_result(conn, hid, "REJECTED", NOW, {"status": "REJECTED_NOT_SIGNIFICANT", "n": 20})

    report = build_knowledge_state_report(conn, now=NOW)
    assert len(report.rejected_hypotheses) == 1
    assert report.rejected_hypotheses[0]["reason"] == "REJECTED_NOT_SIGNIFICANT"


def test_decay_state_tolerates_naive_now_against_aware_stored_timestamps():
    """Regression test: a caller passing a naive datetime.now() (instead
    of datetime.now(timezone.utc)) must not crash against the tz-aware
    timestamps store/db.py always writes - found running the real
    experiment script, which had made exactly this mistake."""
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "X", "direction": "positive"}, 20, 0.05, None, None,
                            20, "ACTIVE", NOW, last_revalidated_at=NOW)
    naive_now = datetime(2024, 6, 5)  # no tzinfo
    report = build_knowledge_state_report(conn, now=naive_now)  # must not raise
    assert report.active_relationships[0].decay_state == "FRESH"


def test_knowledge_version_reflects_registry_entry_count():
    conn = db.connect(":memory:")
    report_before = build_knowledge_state_report(conn, now=NOW)
    assert report_before.knowledge_version == 0

    db.register_change(conn, "v1", "test", {}, None, None, {}, "test-suite", "PROMOTED", NOW)
    report_after = build_knowledge_state_report(conn, now=NOW)
    assert report_after.knowledge_version == 1


def test_llm_status_defaults_to_rule_based_when_env_unset(monkeypatch):
    monkeypatch.delenv("INTERPRETER_PROVIDER", raising=False)
    monkeypatch.delenv("HYPOTHESIS_PROVIDER", raising=False)
    conn = db.connect(":memory:")
    report = build_knowledge_state_report(conn, now=NOW)
    assert "INTERPRETER_PROVIDER=rule_based" in report.llm_status
    assert "HYPOTHESIS_PROVIDER=rule_based" in report.llm_status
    assert "NO LLM reasoning is occurring" in report.llm_status


def test_llm_status_reflects_explicit_env_override(monkeypatch):
    monkeypatch.setenv("INTERPRETER_PROVIDER", "llm")
    monkeypatch.setenv("HYPOTHESIS_PROVIDER", "rule_based")
    conn = db.connect(":memory:")
    report = build_knowledge_state_report(conn, now=NOW)
    assert "INTERPRETER_PROVIDER=llm" in report.llm_status
    assert "NO LLM reasoning is occurring" not in report.llm_status


def test_operational_counts_reflects_ledger_and_governance_state():
    conn = db.connect(":memory:")
    event = EventRecord("A", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                         {"regime": "RISK_OFF"})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
    resolved_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, resolved_id, -0.03, NOW, -0.01, "OK")

    unresolved_event = EventRecord("B", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                                    {"regime": "RISK_OFF"})
    unresolved_pred = PredictionRecord(20, None, "INSUFFICIENT_PRECEDENT", {"basis": "no_baseline_for_horizon"},
                                        "STATIC_v1", NOW)
    db.log_prediction(conn, unresolved_event, unresolved_pred)

    hid = db.add_hypothesis(conn, resolved_id, {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                             "explanation", NOW)
    db.set_hypothesis_result(conn, hid, "CONFIRMED", NOW, {"status": "CONFIRMED", "n": 20})

    report = build_knowledge_state_report(conn, now=NOW)
    counts = report.operational_counts
    assert counts.n_events_ingested == 2
    assert counts.n_events_resolved == 1
    assert counts.n_events_insufficient_precedent == 1
    assert counts.error_type_distribution == {"OK": 1}
    assert counts.n_hypotheses_generated == 1
    assert counts.n_hypotheses_confirmed == 1
    assert counts.n_hypotheses_rejected == 0


def test_concepts_section_covers_all_ontology_concepts_even_untested_ones():
    conn = db.connect(":memory:")
    report = build_knowledge_state_report(conn, now=NOW)
    concepts_seen = {c.concept for c in report.concepts}
    assert len(report.concepts) == 22  # stage 7 item 7 added CLOSE_LOCATION_VALUE, LIQUIDITY_REGIME
    assert "BREAKOUT" in concepts_seen
    breakout = next(c for c in report.concepts if c.concept == "BREAKOUT")
    assert breakout.active_relationships == []
    assert breakout.computable is True
    sector = next(c for c in report.concepts if c.concept == "SECTOR_CONTEXT")
    assert sector.computable is False


def test_concept_summary_places_active_relationship_correctly():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.10, -0.15, -0.05, 30,
                            "ACTIVE", NOW, concept="BREAKOUT", methodology_ids=["meth-1"])
    report = build_knowledge_state_report(conn, now=NOW)
    breakout = next(c for c in report.concepts if c.concept == "BREAKOUT")
    assert len(breakout.active_relationships) == 1
    assert breakout.active_relationships[0].relationship_id == "rel-1"
    assert breakout.active_relationships[0].methodology_ids == ["meth-1"]
    assert breakout.decaying_relationships == []


def test_concept_summary_untested_and_rejected_hypotheses():
    conn = db.connect(":memory:")
    event = EventRecord("A", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                         {"regime": "RISK_OFF"})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
    eid = db.log_prediction(conn, event, pred)
    hid_untested = db.add_hypothesis(conn, eid, {"event_type": "GUIDANCE_CHANGE", "breakout_state": "BREAKOUT_UP"},
                                      20, "explanation", NOW, concept="BREAKOUT")
    hid_rejected = db.add_hypothesis(conn, eid, {"event_type": "GUIDANCE_CHANGE", "momentum_state": "POSITIVE"},
                                      20, "explanation", NOW, concept="MOMENTUM")
    db.set_hypothesis_result(conn, hid_rejected, "REJECTED", NOW, {"status": "REJECTED_NOT_SIGNIFICANT", "n": 20})

    report = build_knowledge_state_report(conn, now=NOW)
    breakout = next(c for c in report.concepts if c.concept == "BREAKOUT")
    assert len(breakout.untested_hypotheses) == 1
    assert breakout.untested_hypotheses[0].hypothesis_id == hid_untested

    momentum = next(c for c in report.concepts if c.concept == "MOMENTUM")
    assert len(momentum.rejected_hypotheses) == 1
    assert momentum.rejected_hypotheses[0].n == 20


def test_decaying_relationship_flagged_without_changing_its_stored_status():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.10, None, None, 30,
                            "ACTIVE", NOW, concept="BREAKOUT")

    def _log_and_resolve(entity, error_type):
        event = EventRecord(entity, "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                             {"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.10, "HIGH", {"basis": "validated_relationship", "relationship_id": "rel-1"},
                                 "ADAPTIVE_v1", NOW)
        eid = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, eid, 0.03, NOW, 0.13, error_type)

    # 6 predictions used this relationship, 4 turned out wrong (>50%, above DECAY_WARNING_MIN_N=5... need >=5)
    for i in range(4):
        _log_and_resolve(f"WRONG{i}", "WRONG_DIRECTION")
    for i in range(2):
        _log_and_resolve(f"RIGHT{i}", "OK")

    report = build_knowledge_state_report(conn, now=NOW)
    breakout = next(c for c in report.concepts if c.concept == "BREAKOUT")
    assert len(breakout.decaying_relationships) == 1
    assert breakout.decaying_relationships[0].relationship_id == "rel-1"
    assert breakout.active_relationships == []

    # the raw stored status is untouched - DECAYING is a report classification only
    raw = conn.execute("SELECT status FROM validated_relationships WHERE relationship_id = 'rel-1'").fetchone()
    assert raw["status"] == "ACTIVE"


def test_active_relationship_below_decay_threshold_is_not_flagged():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.10, None, None, 30,
                            "ACTIVE", NOW, concept="BREAKOUT")

    def _log_and_resolve(entity, error_type):
        event = EventRecord(entity, "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                             {"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.10, "HIGH", {"basis": "validated_relationship", "relationship_id": "rel-1"},
                                 "ADAPTIVE_v1", NOW)
        eid = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, eid, -0.11, NOW, -0.01, error_type)

    for i in range(6):
        _log_and_resolve(f"RIGHT{i}", "OK")  # all correct - well below the 50% contradiction threshold

    report = build_knowledge_state_report(conn, now=NOW)
    breakout = next(c for c in report.concepts if c.concept == "BREAKOUT")
    assert len(breakout.active_relationships) == 1
    assert breakout.decaying_relationships == []


def test_methodology_summary_reports_active_and_no_active_evidence_concepts():
    conn = db.connect(":memory:")
    db.add_methodology(conn, "meth-1", "Test System", "Test Trader", "book", "desc", "RULE_BASED", NOW)
    db.add_methodology_concept_link(conn, "link-1", "meth-1", "BREAKOUT", "rationale", NOW)
    db.add_methodology_concept_link(conn, "link-2", "meth-1", "MOMENTUM", "rationale", NOW)
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.10, None, None, 30,
                            "ACTIVE", NOW, concept="BREAKOUT")

    report = build_knowledge_state_report(conn, now=NOW)
    assert len(report.methodologies) == 1
    meth = report.methodologies[0]
    assert meth.concepts_claimed == ["BREAKOUT", "MOMENTUM"]
    assert meth.concepts_with_active_evidence == ["BREAKOUT"]
    assert meth.concepts_with_no_active_evidence == ["MOMENTUM"]


def test_methodology_summary_with_no_active_concepts_at_all():
    conn = db.connect(":memory:")
    db.add_methodology(conn, "meth-1", "Test System", "Test Trader", "book", "desc", "RULE_BASED", NOW)
    db.add_methodology_concept_link(conn, "link-1", "meth-1", "PULLBACK", "rationale", NOW)
    report = build_knowledge_state_report(conn, now=NOW)
    meth = report.methodologies[0]
    assert meth.concepts_with_active_evidence == []
    assert meth.concepts_with_no_active_evidence == ["PULLBACK"]


def test_calibration_by_horizon_splits_static_and_adaptive():
    conn = db.connect(":memory:")

    def _log(model_version, horizon, predicted, realized):
        event = EventRecord("A", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                             {"regime": "RISK_OFF"})
        pred = PredictionRecord(horizon, predicted, "MEDIUM", {"basis": "unconditional_baseline"},
                                 model_version, NOW)
        eid = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, eid, realized, NOW, realized - predicted, "OK")

    _log("STATIC_v1", 20, -0.02, -0.03)
    _log("ADAPTIVE_v1", 20, -0.05, -0.03)

    report = build_knowledge_state_report(conn, now=NOW)
    by_horizon = {c.horizon_days: c for c in report.calibration_by_horizon}
    assert by_horizon[20].static_n == 1
    assert by_horizon[20].adaptive_n == 1
