from datetime import datetime, timedelta, timezone

import pytest

from market_agent.agents.adaptive_agent import AdaptiveAgent
from market_agent.agents.static_agent import StaticAgent
from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.events.schema import ContextSnapshot, EventRecord, PredictionRecord, RawItem
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.pipeline import interpret_event, predict_event, predict_for_security, predict_security
from market_agent.pit.clock import FutureInformationError, PointInTimeClock
from market_agent.store import db

BASELINE = {20: 0.02}
NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
CTX = ContextSnapshot(regime="RISK_OFF", prior_5d_return=-0.03, sector_momentum="UNKNOWN")


def test_interpret_event_rejects_future_timestamped_item():
    clock = PointInTimeClock(now=NOW)
    future_item = RawItem("cuts guidance", "wire", "AAPL", NOW + timedelta(days=1))
    with pytest.raises(FutureInformationError):
        interpret_event(RuleBasedInterpreter(), future_item, CTX, clock)


def test_interpret_event_accepts_past_item():
    clock = PointInTimeClock(now=NOW)
    item = RawItem("cuts guidance", "wire", "AAPL", NOW - timedelta(days=1))
    event = interpret_event(RuleBasedInterpreter(), item, CTX, clock)
    assert event is not None
    assert event.event_type == "GUIDANCE_CHANGE"


def _event(entity="AAPL", context=None):
    return EventRecord(entity, "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", NOW, NOW,
                        context or CTX.to_dict())


def test_no_precedent_at_all_reports_insufficient_precedent():
    conn = db.connect(":memory:")  # empty store - zero similar cases can exist
    static = StaticAgent(BASELINE)
    results = predict_for_security(conn, static, _event(), [20], prices=None, predicted_at=NOW)
    assert len(results) == 1
    assert results[0].status == "INSUFFICIENT_PRECEDENT"
    assert results[0].n_similar_cases == 0
    assert results[0].novelty_score == 1.0


def test_precedent_present_reports_ok_with_reasoning():
    conn = db.connect(":memory:")
    # log one resolved, matching precedent case
    prior_event = EventRecord("PEER", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance",
                               NOW - timedelta(days=100), NOW - timedelta(days=100), CTX.to_dict())
    prior_pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1",
                                   NOW - timedelta(days=100))
    prior_id = db.log_prediction(conn, prior_event, prior_pred)
    db.record_outcome(conn, prior_id, -0.05, NOW - timedelta(days=80), -0.03, "WRONG_MAGNITUDE")

    static = StaticAgent(BASELINE)
    results = predict_for_security(conn, static, _event(), [20], prices=None, predicted_at=NOW)
    assert results[0].status == "OK"
    assert results[0].n_similar_cases == 1
    assert "PEER" not in results[0].reasoning_provenance[0]  # sanity: reasoning is prose, not a leak of raw ids
    assert any("similar historical case" in r for r in results[0].reasoning_provenance)


def test_validated_relationship_basis_reports_ci_derived_uncertainty():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, -0.11, -0.07, 40, "ACTIVE", NOW)
    adaptive = AdaptiveAgent(conn, BASELINE)
    results = predict_for_security(conn, adaptive, _event(), [20], prices=None, predicted_at=NOW)
    assert results[0].status == "OK"
    assert results[0].basis["basis"] == "validated_relationship"
    assert results[0].uncertainty == pytest.approx(0.02)  # (−0.07 − (−0.11)) / 2


def test_no_baseline_for_horizon_is_insufficient_precedent():
    conn = db.connect(":memory:")
    static = StaticAgent({})  # no baseline for any horizon
    results = predict_for_security(conn, static, _event(), [20], prices=None, predicted_at=NOW)
    assert results[0].status == "INSUFFICIENT_PRECEDENT"
    assert results[0].predicted_impact is None


def test_multiple_horizons_are_independent_results():
    conn = db.connect(":memory:")
    static = StaticAgent({20: 0.02, 60: 0.05})
    results = predict_for_security(conn, static, _event(), [20, 60], prices=None, predicted_at=NOW)
    assert len(results) == 2
    assert {r.horizon_days for r in results} == {20, 60}


class FlatPrices(PriceSeriesProvider):
    """Every ticker flat at 100 (benchmark included) - deterministic,
    network-free: NORMAL regime, zero trailing returns."""

    def close_price(self, ticker, as_of):
        return 100.0


# --- item 10: predict_event(event) ---

def test_predict_event_not_recognized_for_out_of_scope_text():
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL announces a new product", "wire", NOW, NOW, [20])
    assert result.status == "NOT_RECOGNIZED"
    assert result.event is None
    assert result.predictions == []


def test_predict_event_ok_composes_interpret_context_and_predict():
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL corp cuts guidance", "wire", NOW, NOW, [20])
    assert result.status == "OK"
    assert result.event.event_type == "GUIDANCE_CHANGE"
    assert result.event.direction == "negative"
    assert result.event.context.get("regime") == "NORMAL"  # flat prices -> zero trailing return
    assert len(result.predictions) == 1
    assert result.predictions[0].horizon_days == 20


def test_predict_event_does_not_log_to_the_ledger():
    """predict_event is a pure query, same contract as predict_for_security -
    the caller decides whether to persist (see market_agent/predict.py)."""
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL", "AAPL corp cuts guidance",
                   "wire", NOW, NOW, [20])
    assert conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"] == 0


# --- item 10: predict_security(security, context) ---

def test_predict_security_with_no_active_relationships_reports_regime_only():
    conn = db.connect(":memory:")
    outlook = predict_security(conn, "AAPL", FlatPrices(), NOW)
    assert outlook.entity == "AAPL"
    assert outlook.regime == "NORMAL"
    assert outlook.applicable_relationships == []
    assert outlook.recent_predictions == []


def test_predict_security_surfaces_matching_active_relationship():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "NORMAL"}, 20, -0.06, -0.09, -0.03, 30, "ACTIVE", NOW)
    # a relationship keyed to a DIFFERENT regime must not show up
    db.upsert_relationship(conn, "rel-2", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.10, None, None, 20, "ACTIVE", NOW)

    outlook = predict_security(conn, "AAPL", FlatPrices(), NOW)
    ids = {r.relationship_id for r in outlook.applicable_relationships}
    assert ids == {"rel-1"}


def test_predict_security_surfaces_recent_predictions_from_the_ledger():
    conn = db.connect(":memory:")
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", NOW)
    db.log_prediction(conn, _event(entity="AAPL"), pred)

    outlook = predict_security(conn, "AAPL", FlatPrices(), NOW)
    assert len(outlook.recent_predictions) == 1
    assert outlook.recent_predictions[0]["predicted_impact"] == -0.02
