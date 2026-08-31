"""End-to-end proof of the full loop from the blueprint:
observe -> predict -> record -> observe outcome -> diagnose -> hypothesis
-> test -> shadow probation -> promote to ACTIVE -> predict better next
time - and, just as important, proof that Agent-STATIC stays completely
frozen throughout.

This is the single test that stands in for "did we actually build genuine
learning, or fake self-modification." If Agent-ADAPTIVE's second
prediction doesn't move toward the realized pattern, or Agent-STATIC's
does move at all, the loop is broken regardless of what any smaller unit
test says.
"""
from datetime import datetime, timedelta, timezone

from market_agent.agents.adaptive_agent import AdaptiveAgent
from market_agent.agents.static_agent import StaticAgent
from market_agent.events.schema import ContextSnapshot, EventRecord, PredictionRecord
from market_agent.learn.error_taxonomy import classify_error
from market_agent.learn.governance import apply_test_results
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator, formalize_and_store
from market_agent.learn.hypothesis_testing import test_hypotheses_batch as run_batch
from market_agent.learn.shadow import evaluate_shadow_relationships
from market_agent.store import db

BASELINE = {20: 0.02}
SEED_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
TRIGGER_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)
SECOND_EVENT_DATE = datetime(2024, 9, 1, tzinfo=timezone.utc)


def _seed_background_history(conn, n=20, effect=-0.10):
    """N already-resolved historical events, all in RISK_OFF, all showing
    a consistent -10% reaction - the real, pre-existing pattern the
    hypothesis test should discover. Published well before TRIGGER_DATE."""
    for i in range(n):
        published = SEED_START + timedelta(days=10 * i)
        event = EventRecord(entity=f"PEER{i}", event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                             source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                             ingested_at=published, context={"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
        event_id = db.log_prediction(conn, event, pred)
        jitter = 0.005 if i % 2 == 0 else -0.005
        db.record_outcome(conn, event_id, effect + jitter, published + timedelta(days=20),
                           (effect + jitter) - (-0.02), "WRONG_MAGNITUDE")


def _predict_and_log(conn, agent, entity, published):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                         ingested_at=published, context=ContextSnapshot("RISK_OFF", -0.05, "NEGATIVE").to_dict())
    prediction = agent.predict(event, horizon_days=20, predicted_at=published)
    event_id = db.log_prediction(conn, event, prediction)
    return event_id, prediction


def test_full_observe_predict_learn_predict_better_loop():
    conn = db.connect(":memory:")
    _seed_background_history(conn)

    static = StaticAgent(BASELINE)
    adaptive = AdaptiveAgent(conn, BASELINE)

    # --- round 1: before any learning, both agents must agree ---
    static_event_id_1, static_pred_1 = _predict_and_log(conn, static, "NVDA", TRIGGER_DATE)
    adaptive_event_id_1, adaptive_pred_1 = _predict_and_log(conn, adaptive, "NVDA", TRIGGER_DATE)
    assert static_pred_1.predicted_impact == adaptive_pred_1.predicted_impact == -0.02

    # --- outcome arrives: reality matches the seeded RISK_OFF pattern, not the naive baseline ---
    realized = -0.095
    error = classify_error(adaptive_pred_1.predicted_impact, adaptive_pred_1.predicted_confidence, realized, "OK")
    assert error.error_type == "WRONG_MAGNITUDE"
    assert error.may_learn_from is True

    db.record_outcome(conn, static_event_id_1, realized, TRIGGER_DATE + timedelta(days=20),
                       error.error_value, error.error_type)
    db.record_outcome(conn, adaptive_event_id_1, realized, TRIGGER_DATE + timedelta(days=20),
                       error.error_value, error.error_type)

    # --- diagnose -> hypothesis -> test -> promote (only from the ADAPTIVE-side event, matching the
    #     blueprint: hypotheses arise from investigating a prediction, not from a second parallel path) ---
    # The event's context has prior_5d_return=-0.05 set (see _predict_and_log), so the generator's
    # bounded powerset (learn/hypothesis.py) proposes THREE candidates from the 2 available
    # dimensions: regime alone, prior_return_bucket alone, and the two combined. Only the
    # regime-alone one can be confirmed here - the seeded background history was logged with no
    # prior_5d_return in its context at all (see _seed_background_history), so it has no
    # prior_return_bucket to match the other two candidates against, and both correctly get
    # REJECTED_INSUFFICIENT_N (N=0).
    adaptive_row = db.get_event(conn, adaptive_event_id_1)
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), adaptive_row, error.error_type,
                                horizon_days=20, proposed_at=TRIGGER_DATE + timedelta(days=21))
    assert len(hids) == 3

    hyp_rows = [conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (h,)).fetchone()
                for h in hids]
    results = run_batch(conn, hyp_rows, BASELINE)
    confirmed = [r for r in results if r.status == "CONFIRMED"]
    assert len(confirmed) == 1
    rejected = [r for r in results if r.status != "CONFIRMED"]
    assert len(rejected) == 2 and all(r.status == "REJECTED_INSUFFICIENT_N" for r in rejected)

    shadow_start = TRIGGER_DATE + timedelta(days=22)
    # Pass the FULL results list (both confirmed and rejected) - apply_test_results handles both:
    # it promotes the confirmed one to shadow AND marks the rejected hypothesis's own row REJECTED,
    # matching how the walkforward harness actually calls it.
    promoted = apply_test_results(conn, results, promoted_by="test-suite", clock_now=shadow_start)
    assert len(promoted) == 1
    relationship_id = promoted[0]
    rel = conn.execute("SELECT status FROM validated_relationships WHERE relationship_id = ?",
                        (relationship_id,)).fetchone()
    assert rel["status"] == "SHADOW"  # confirmed, but not yet live - see learn/shadow.py

    # --- shadow probation: 10 NEW, disjoint RISK_OFF/negative observations accumulate before the
    #     round-2 prediction - without this, the relationship would still be in SHADOW and round 2
    #     would fall back to the unconditional baseline, same as round 1. This is the mechanism
    #     working as designed, not a shortcut to make the test pass. ---
    for i in range(10):
        published = shadow_start + timedelta(days=5 * (i + 1))
        event = EventRecord(entity=f"SHADOWPEER{i}", event_type="GUIDANCE_CHANGE", direction="negative",
                             source="wire", source_reliability_snapshot=0.5, raw_text="cuts guidance",
                             published_at=published, ingested_at=published, context={"regime": "RISK_OFF"})
        pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
        event_id = db.log_prediction(conn, event, pred)
        jitter = 0.004 if i % 2 == 0 else -0.004
        db.record_outcome(conn, event_id, -0.10 + jitter, published + timedelta(days=20),
                           (-0.10 + jitter) - (-0.02), "WRONG_MAGNITUDE")

    shadow_summary = evaluate_shadow_relationships(conn, BASELINE, promoted_by="test-suite",
                                                     clock_now=SECOND_EVENT_DATE)
    assert any(s["relationship_id"] == relationship_id and s["outcome"] == "ACTIVE" for s in shadow_summary), \
        f"expected the relationship to graduate SHADOW->ACTIVE on new evidence, got: {shadow_summary}"

    # --- round 2: a DIFFERENT entity, same RISK_OFF guidance-cut situation, months later ---
    static_pred_2 = static.predict(
        EventRecord("MSFT", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", SECOND_EVENT_DATE,
                    SECOND_EVENT_DATE, ContextSnapshot("RISK_OFF", -0.05, "NEGATIVE").to_dict()),
        20, SECOND_EVENT_DATE)
    adaptive_pred_2 = adaptive.predict(
        EventRecord("MSFT", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", SECOND_EVENT_DATE,
                    SECOND_EVENT_DATE, ContextSnapshot("RISK_OFF", -0.05, "NEGATIVE").to_dict()),
        20, SECOND_EVENT_DATE)

    # Agent-STATIC: completely unchanged - frozen, exactly as required.
    assert static_pred_2.predicted_impact == static_pred_1.predicted_impact == -0.02

    # Agent-ADAPTIVE: prediction moved, and moved toward what actually keeps happening in this
    # regime (~-0.10), not just moved arbitrarily.
    assert adaptive_pred_2.predicted_impact != adaptive_pred_1.predicted_impact
    assert adaptive_pred_2.predicted_impact < -0.07  # materially closer to the real -0.095/-0.10 pattern
    assert adaptive_pred_2.basis["basis"] == "validated_relationship"

    # And critically: this is a DIFFERENT entity (MSFT) than the one the hypothesis was generated
    # from (NVDA) - this is generalization to a structurally similar future situation, not the
    # system simply retrieving and repeating the one case it saw.
    assert adaptive_pred_2.basis.get("relationship_id") is not None
