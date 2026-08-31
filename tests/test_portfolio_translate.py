from datetime import datetime, timedelta, timezone

from market_agent.agents.static_agent import StaticAgent
from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.pipeline import predict_event
from market_agent.portfolio.translate import translate_event_to_portfolio
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
BASELINE = {20: 0.02}


class FlatPrices(PriceSeriesProvider):
    def close_price(self, ticker, as_of):
        return 100.0


def test_no_recognized_event_yields_no_portfolio_impact():
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL launches a new store", "wire", NOW, NOW, [20])
    report = translate_event_to_portfolio(conn, FlatPrices(), {"AAPL": 0.5}, result, 20, NOW)
    assert report.portfolio_expected_impact is None
    assert report.holdings == []


def _seed_similar_resolved_case(conn):
    """One prior, resolved GUIDANCE_CHANGE/negative case in the same
    regime/prior-return bucket FlatPrices produces (NORMAL/FLAT) - without
    this, predict_for_security correctly downgrades the unconditional
    baseline to INSUFFICIENT_PRECEDENT (zero retrieved precedent), which
    is real, intended behavior (see pipeline.py's own docstring), not
    something to work around in every test."""
    from market_agent.events.schema import EventRecord, PredictionRecord
    published = NOW - timedelta(days=10)
    event = EventRecord("PEER", "GUIDANCE_CHANGE", "negative", "wire", 0.5, "cuts guidance", published, published,
                         {"regime": "NORMAL", "prior_5d_return": 0.0})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, -0.025, published + timedelta(days=20), -0.005, "OK")


def test_directly_affected_holding_gets_weighted_contribution():
    conn = db.connect(":memory:")
    _seed_similar_resolved_case(conn)
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL corp cuts guidance", "wire", NOW, NOW, [20])
    assert result.status == "OK"

    report = translate_event_to_portfolio(conn, FlatPrices(), {"AAPL": 0.4, "MSFT": 0.6}, result, 20, NOW)

    aapl = next(h for h in report.holdings if h.entity == "AAPL")
    msft = next(h for h in report.holdings if h.entity == "MSFT")

    assert aapl.is_directly_affected is True
    assert aapl.status == "DIRECT_EVENT"
    assert aapl.predicted_impact == result.predictions[0].predicted_impact
    assert abs(aapl.weighted_contribution - (0.4 * aapl.predicted_impact)) < 1e-9

    assert msft.is_directly_affected is False
    assert msft.status == "NO_DIRECT_EVENT"
    assert msft.predicted_impact is None
    assert msft.weighted_contribution is None

    assert report.n_holdings_with_data == 1
    assert abs(report.portfolio_expected_impact - aapl.weighted_contribution) < 1e-9


def test_triggering_entity_not_held_has_no_portfolio_impact_but_still_reports_other_holdings():
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "NVDA",
                            "NVDA corp cuts guidance", "wire", NOW, NOW, [20])
    report = translate_event_to_portfolio(conn, FlatPrices(), {"MSFT": 1.0}, result, 20, NOW)
    assert report.portfolio_expected_impact is None
    assert report.holdings[0].entity == "MSFT"
    assert report.holdings[0].status == "NO_DIRECT_EVENT"
    assert report.n_holdings_with_data == 0


def test_no_baseline_for_horizon_reports_insufficient_precedent_not_zero():
    conn = db.connect(":memory:")
    static = StaticAgent({})  # no baseline for horizon 20 at all
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL corp cuts guidance", "wire", NOW, NOW, [20])
    report = translate_event_to_portfolio(conn, FlatPrices(), {"AAPL": 1.0}, result, 20, NOW)
    aapl = report.holdings[0]
    assert aapl.status == "INSUFFICIENT_PRECEDENT"
    assert aapl.predicted_impact is None
    assert report.portfolio_expected_impact is None


def test_portfolio_dict_is_never_persisted():
    conn = db.connect(":memory:")
    static = StaticAgent(BASELINE)
    result = predict_event(conn, static, FlatPrices(), RuleBasedInterpreter(), "AAPL",
                            "AAPL corp cuts guidance", "wire", NOW, NOW, [20])
    n_events_before = conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"]
    n_rel_before = conn.execute("SELECT COUNT(*) c FROM validated_relationships").fetchone()["c"]
    translate_event_to_portfolio(conn, FlatPrices(), {"AAPL": 0.9, "TSLA": 0.1}, result, 20, NOW)
    assert conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"] == n_events_before
    assert conn.execute("SELECT COUNT(*) c FROM validated_relationships").fetchone()["c"] == n_rel_before
