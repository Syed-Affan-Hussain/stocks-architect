from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.hierarchical_research import (
    DEFAULT_RESEARCH_BUDGET, ResearchBudget, run_hierarchical_research_pass, run_level1_family_screening,
)
from market_agent.store import db

BASELINE = {20: 0.02}
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2024, 12, 1, tzinfo=timezone.utc)


def _log_and_resolve(conn, entity, breakout_state, regime, realized, i):
    published = START + timedelta(days=3 * i)
    context = {"regime": regime, "breakout_state": breakout_state}
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                         source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                         ingested_at=published, context=context)
    pred = PredictionRecord(20, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized, published + timedelta(days=20), realized - 0.02, "OK")


def test_level1_rejects_family_with_insufficient_n():
    conn = db.connect(":memory:")
    for i in range(5):  # below MIN_N=15
        _log_and_resolve(conn, f"E{i}", "BREAKOUT_UP", "RISK_ON", 0.10, i)
    results, dropped = run_level1_family_screening(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(),
                                                     BASELINE, DEFAULT_RESEARCH_BUDGET)
    breakout = next(r for r in results if r.concept == "BREAKOUT")
    assert breakout.test_result.status == "REJECTED_INSUFFICIENT_N"
    assert dropped == []


def test_level1_confirms_family_with_a_real_strong_effect():
    conn = db.connect(":memory:")
    for i in range(30):
        jitter = 0.005 if i % 2 == 0 else -0.005
        _log_and_resolve(conn, f"E{i}", "BREAKOUT_UP", "RISK_ON", 0.10 + jitter, i)
    results, _ = run_level1_family_screening(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(),
                                               BASELINE, DEFAULT_RESEARCH_BUDGET)
    breakout = next(r for r in results if r.concept == "BREAKOUT")
    assert breakout.test_result.status == "CONFIRMED"
    assert breakout.test_result.n == 30


def test_budget_drops_families_beyond_the_cap():
    conn = db.connect(":memory:")
    small_budget = ResearchBudget(max_level1_families=2, max_level2_setups_per_family=4,
                                   max_level3_context_dims_per_setup=3, label="test_small")
    results, dropped = run_level1_family_screening(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(),
                                                     BASELINE, small_budget)
    assert len(results) == 2
    assert len(dropped) == 16  # 18 technical dimensions with a state field - 2 tested


def test_full_pass_confirms_setup_and_runs_context_conditioning():
    conn = db.connect(":memory:")
    for i in range(30):
        jitter = 0.005 if i % 2 == 0 else -0.005
        _log_and_resolve(conn, f"E{i}", "BREAKOUT_UP", "RISK_ON", 0.10 + jitter, i)

    report = run_hierarchical_research_pass(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(), BASELINE,
                                             proposed_at=NOW, promoted_by="test-suite")

    breakout = next(r for r in report.level1_results if r.concept == "BREAKOUT")
    assert breakout.test_result.status == "CONFIRMED"

    assert "breakout_state" in report.level2_results
    setup_results = report.level2_results["breakout_state"]
    assert len(setup_results) >= 1
    confirmed_setup = next((r for r in setup_results if r.test_result.status == "CONFIRMED"), None)
    assert confirmed_setup is not None
    assert confirmed_setup.incremental_value is not None  # the diagnostic ran alongside the real test

    # a real, SHADOW-status relationship should now exist, tagged with the concept
    rel = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id IN "
                        "(SELECT relationship_id FROM validated_relationships WHERE concept = 'BREAKOUT')").fetchone()
    assert rel is not None
    assert rel["status"] == "SHADOW"

    # Level 3 should have run for the confirmed setup (regime=RISK_ON is the only value present here)
    assert any(k.startswith("breakout_state=BREAKOUT_UP") for k in report.level3_results)


def test_family_that_fails_level1_never_spawns_level2_candidates():
    conn = db.connect(":memory:")
    # only 5 rows - insufficient N, family will be rejected at Level 1
    for i in range(5):
        _log_and_resolve(conn, f"E{i}", "BREAKOUT_UP", "RISK_ON", 0.10, i)

    report = run_hierarchical_research_pass(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(), BASELINE,
                                             proposed_at=NOW, promoted_by="test-suite")
    assert "breakout_state" not in report.level2_results
    n_hypotheses = conn.execute("SELECT COUNT(*) c FROM candidate_hypotheses").fetchone()["c"]
    assert n_hypotheses == 0  # Level 1 never writes a candidate_hypotheses row


def test_evidence_records_the_budget_before_any_test_runs():
    conn = db.connect(":memory:")
    report = run_hierarchical_research_pass(conn, "GUIDANCE_CHANGE", "positive", 20, NOW.isoformat(), BASELINE,
                                             proposed_at=NOW, promoted_by="test-suite")
    assert "ResearchBudget recorded before execution" in report.evidence[0]
