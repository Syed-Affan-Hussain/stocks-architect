from datetime import datetime, timedelta, timezone

from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.research.evaluation.outcome_resolution import resolve_outcomes
from market_agent.store import db

TRIGGERED = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakePrices(PriceSeriesProvider):
    def __init__(self, prices: dict):
        self.prices = prices

    def close_price(self, ticker, as_of):
        return self.prices.get((ticker, as_of.date().isoformat()))


def _log_one(conn, entity="ACME", triggered_at=TRIGGERED):
    return db.save_prediction(conn, entity=entity, mode="A_NO_NEWS", triggered_at=triggered_at,
                               model_version="test_v1", decision_label="FAVORABLE", predicted_impact=1.0,
                               predicted_confidence=0.7, inputs_snapshot={"entity": entity})


def test_horizon_not_yet_matured_is_left_untouched():
    conn = db.connect(":memory:")
    _log_one(conn)
    prices = FakePrices({})  # no prices needed - should never even be queried
    resolved = resolve_outcomes(conn, prices, now=TRIGGERED + timedelta(hours=1))
    assert resolved == []
    row = db.all_predictions(conn)[0]
    assert row["realized_return_1d"] is None


def test_matured_1d_horizon_is_resolved_with_real_prices():
    conn = db.connect(":memory:")
    _log_one(conn)
    prices = FakePrices({
        ("ACME", "2024-01-01"): 100.0, ("ACME", "2024-01-02"): 103.0,
        ("SPY", "2024-01-01"): 400.0, ("SPY", "2024-01-02"): 402.0,
    })
    now = TRIGGERED + timedelta(days=2)  # well past the 1-trading-day (1 calendar day) horizon
    resolved = resolve_outcomes(conn, prices, now=now)
    assert len(resolved) == 1
    assert resolved[0]["horizon_trading_days"] == 1
    row = db.all_predictions(conn)[0]
    assert row["realized_return_1d"] is not None
    assert row["resolved_1d_at"] is not None
    # 3% entity move minus 0.5% benchmark move = 2.5% abnormal
    assert abs(row["realized_return_1d"] - 0.025) < 1e-9


def test_missing_price_data_leaves_the_row_null_not_a_guess():
    conn = db.connect(":memory:")
    _log_one(conn)
    prices = FakePrices({})  # nothing available
    resolved = resolve_outcomes(conn, prices, now=TRIGGERED + timedelta(days=2))
    assert resolved == []
    row = db.all_predictions(conn)[0]
    assert row["realized_return_1d"] is None


def test_resolution_is_idempotent_across_repeated_calls():
    conn = db.connect(":memory:")
    _log_one(conn)
    prices = FakePrices({
        ("ACME", "2024-01-01"): 100.0, ("ACME", "2024-01-02"): 100.0,
        ("SPY", "2024-01-01"): 400.0, ("SPY", "2024-01-02"): 400.0,
    })
    now = TRIGGERED + timedelta(days=2)
    first = resolve_outcomes(conn, prices, now=now)
    second = resolve_outcomes(conn, prices, now=now)
    assert len(first) == 1
    assert len(second) == 0  # already resolved - not re-resolved, not double-counted


def test_only_matured_horizons_resolve_others_stay_pending():
    """At day 10, only the 1d and 5d horizons (calendar approximations 1
    and 7 days) have matured - 20d (29 cal days) and 60d (88 cal days)
    must stay untouched."""
    conn = db.connect(":memory:")
    _log_one(conn)
    prices = FakePrices({
        # 1d horizon looks up trigger+1 calendar day; 5d looks up trigger+7 - both real, resolvable data
        ("ACME", "2024-01-01"): 100.0, ("ACME", "2024-01-02"): 101.0, ("ACME", "2024-01-08"): 105.0,
        ("SPY", "2024-01-01"): 400.0, ("SPY", "2024-01-02"): 400.0, ("SPY", "2024-01-08"): 400.0,
    })
    now = TRIGGERED + timedelta(days=10)  # 20d (needs +29d) and 60d (needs +88d) haven't matured yet
    resolve_outcomes(conn, prices, now=now)
    row = db.all_predictions(conn)[0]
    assert row["realized_return_1d"] is not None
    assert row["realized_return_5d"] is not None
    assert row["realized_return_20d"] is None
    assert row["realized_return_60d"] is None
