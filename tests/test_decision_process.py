from datetime import datetime, timezone

from market_agent.strategy.decision_process import (
    build_hypothesis_only_decision_process, build_validated_decision_process,
)
from market_agent.store import db

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
BASELINE = {20: 0.05, 60: 0.08}


def test_validated_decision_process_extracts_setup_and_regime_from_condition():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                                            "breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"},
                            20, 0.06, 0.02, 0.10, 40, "ACTIVE", NOW, concept="BREAKOUT",
                            methodology_ids=["meth-1", "meth-2"])
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()

    dp = build_validated_decision_process(conn, row, BASELINE)
    assert dp is not None
    assert dp.evidence_status == "STATISTICALLY_VALIDATED"
    assert dp.setup.dimension == "breakout_state" and dp.setup.value == "BREAKOUT_UP"
    assert dp.setup.concept == "BREAKOUT"
    assert dp.regime is not None and dp.regime.value == "RISK_ON"
    assert dp.horizon_days == 20
    assert dp.exit.horizon_days == 20  # exit horizon matches the TESTED horizon, never a different one
    assert dp.invalidation.max_adverse_excursion_pct == 2.0 * 0.05  # INVALIDATION_BASELINE_MULTIPLE x baseline
    assert dp.risk.max_position_risk_pct == 0.01
    assert dp.effect_estimate == 0.06
    assert dp.n_supporting == 40
    assert dp.provenance_methodology_ids == ["meth-1", "meth-2"]
    assert dp.source_relationship_id == "rel-1"


def test_validated_decision_process_returns_none_for_pure_event_context_relationship():
    """A relationship conditioned only on regime/prior_return_bucket (no
    technical-concept dimension) has no SETUP in the trading-concept
    sense this decision process represents."""
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 40, "ACTIVE", NOW)
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    dp = build_validated_decision_process(conn, row, BASELINE)
    assert dp is None


def test_validated_decision_process_handles_no_regime_present():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                                            "vwap_state": "ABOVE_VWAP_PROXY"}, 60, 0.03, None, None, 20, "SHADOW",
                            NOW, concept="VWAP")
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    dp = build_validated_decision_process(conn, row, BASELINE)
    assert dp is not None
    assert dp.regime is None
    assert dp.setup.dimension == "vwap_state"
    assert dp.invalidation.max_adverse_excursion_pct == 2.0 * 0.08


def test_validated_decision_process_missing_baseline_leaves_invalidation_none():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                                            "vwap_state": "ABOVE_VWAP_PROXY"}, 5, 0.03, None, None, 20, "SHADOW",
                            NOW, concept="VWAP")
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id='rel-1'").fetchone()
    dp = build_validated_decision_process(conn, row, {})  # no baseline for horizon 5
    assert dp.invalidation.max_adverse_excursion_pct is None


def test_hypothesis_only_decision_process_has_no_evidence_numbers():
    conn = db.connect(":memory:")
    dp = build_hypothesis_only_decision_process(conn, "meth-1", "BREAKOUT", "GUIDANCE_CHANGE", "positive", 20)
    assert dp.evidence_status == "HYPOTHESIS_ONLY"
    assert dp.effect_estimate is None
    assert dp.n_supporting is None
    assert dp.source_relationship_id is None
    assert dp.invalidation.max_adverse_excursion_pct is None
    assert dp.provenance_methodology_ids == ["meth-1"]
