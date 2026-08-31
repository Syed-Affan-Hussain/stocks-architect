from datetime import datetime, timedelta, timezone

from market_agent.outcomes.ohlcv import Bar, OHLCVProvider
from market_agent.strategy.outcome_engine import compute_strategy_outcome_report, compute_trade_outcome

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data: dict[str, list[Bar]] = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff]


def _bar(i, o, h, l, c, v=1_000_000):
    return Bar(date=START + timedelta(days=i), open=o, high=h, low=l, close=c, volume=v)


def test_long_trade_realized_return_and_mfe_mae_from_real_highs_lows():
    ohlcv = FakeOHLCV()
    bars = [
        _bar(0, 100, 101, 99, 100),    # entry bar
        _bar(1, 100, 115, 100, 110),   # a spike to 115 intrabar - MFE should reflect this, not just close=110
        _bar(2, 110, 111, 90, 95),     # a dip to 90 intrabar - MAE should reflect this
        _bar(3, 95, 106, 94, 105),     # exit bar, close=105
    ]
    ohlcv.set_bars("ACME", bars)
    outcome = compute_trade_outcome(ohlcv, "ACME", "LONG", START, horizon_days=3, invalidation_level=-0.20)
    assert outcome is not None
    assert outcome.entry_price == 100
    assert outcome.exit_price == 105
    assert abs(outcome.realized_return - 0.05) < 1e-9
    assert abs(outcome.mfe - 0.15) < 1e-9  # (115-100)/100 - from the intrabar high, not the close
    assert abs(outcome.mae - (-0.10)) < 1e-9  # (90-100)/100 - from the intrabar low, not the close
    assert outcome.hit_invalidation is False  # -0.10 never breached -0.20


def test_short_trade_favorable_and_adverse_directions_are_correctly_inverted():
    ohlcv = FakeOHLCV()
    bars = [
        _bar(0, 100, 101, 99, 100),
        _bar(1, 100, 102, 85, 90),     # a drop to 85 intrabar - FAVORABLE for a short
        _bar(2, 90, 120, 89, 118),     # a spike to 120 intrabar - ADVERSE for a short
        _bar(3, 118, 119, 108, 110),
    ]
    ohlcv.set_bars("ACME", bars)
    outcome = compute_trade_outcome(ohlcv, "ACME", "SHORT", START, horizon_days=3, invalidation_level=-0.05)
    assert outcome is not None
    assert abs(outcome.realized_return - (-0.10)) < 1e-9  # price rose 10% -> a loss for a short
    assert abs(outcome.mfe - 0.15) < 1e-9   # 1 - 85/100 = 0.15, the best point for a short
    assert abs(outcome.mae - (-0.20)) < 1e-9  # 1 - 120/100 = -0.20, the worst point for a short
    assert outcome.hit_invalidation is True  # -0.20 breaches the -0.05 stop


def test_invalidation_day_records_the_first_breach():
    ohlcv = FakeOHLCV()
    bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 101, 80, 85), _bar(2, 85, 86, 60, 65)]
    ohlcv.set_bars("ACME", bars)
    outcome = compute_trade_outcome(ohlcv, "ACME", "LONG", START, horizon_days=2, invalidation_level=-0.10)
    assert outcome.hit_invalidation is True
    assert outcome.invalidation_day == 1  # breached on the SECOND bar (index 1), not the third


def test_insufficient_bars_returns_none():
    ohlcv = FakeOHLCV()
    ohlcv.set_bars("ACME", [_bar(0, 100, 101, 99, 100)])  # only 1 bar
    outcome = compute_trade_outcome(ohlcv, "ACME", "LONG", START, horizon_days=20, invalidation_level=None)
    assert outcome is None


def test_path_returns_records_cumulative_return_at_each_bar():
    ohlcv = FakeOHLCV()
    bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 108, 99, 105), _bar(2, 105, 111, 104, 110)]
    ohlcv.set_bars("ACME", bars)
    outcome = compute_trade_outcome(ohlcv, "ACME", "LONG", START, horizon_days=2, invalidation_level=None)
    assert len(outcome.path_returns) == 3
    assert abs(outcome.path_returns[0] - 0.0) < 1e-9    # entry bar close == entry price
    assert abs(outcome.path_returns[1] - 0.05) < 1e-9   # (105-100)/100
    assert abs(outcome.path_returns[2] - 0.10) < 1e-9   # (110-100)/100


def test_no_invalidation_level_means_never_flagged():
    ohlcv = FakeOHLCV()
    bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 101, 1, 2)]  # a huge drop
    ohlcv.set_bars("ACME", bars)
    outcome = compute_trade_outcome(ohlcv, "ACME", "LONG", START, horizon_days=1, invalidation_level=None)
    assert outcome.hit_invalidation is False


# --- strategy outcome report ---

def _trade(entity, action, realized_return, mfe=0.05, mae=-0.02, confidence="HIGH"):
    from market_agent.strategy.outcome_engine import TradeOutcome
    return TradeOutcome(entity=entity, action=action, entry_date=START.isoformat(), entry_price=100,
                         exit_date=(START + timedelta(days=20)).isoformat(), exit_price=100 * (1 + realized_return),
                         horizon_days=20, holding_days=20, raw_entity_return=realized_return,
                         realized_return=realized_return, mfe=mfe, mae=mae, hit_invalidation=False,
                         invalidation_day=None, confidence=confidence)


def test_report_too_few_trades_returns_none_metrics():
    report = compute_strategy_outcome_report([_trade("A", "LONG", 0.05)], n_decisions_considered=5,
                                              predicted_impacts=[0.05])
    assert report.n_trades == 1
    assert report.win_rate is None


def test_report_win_rate_expectancy_and_profit_factor():
    trades = [_trade(f"E{i}", "LONG", 0.05) for i in range(6)] + [_trade(f"L{i}", "LONG", -0.03) for i in range(4)]
    predicted = [0.05] * 6 + [0.05] * 4
    report = compute_strategy_outcome_report(trades, n_decisions_considered=20, predicted_impacts=predicted,
                                              transaction_cost=0.0)
    assert report.n_trades == 10
    assert report.win_rate == 0.6
    assert abs(report.expectancy - (6 * 0.05 - 4 * 0.03) / 10) < 1e-9
    assert report.profit_factor is not None
    assert report.exposure == 10 / 20


def test_report_hit_rate_by_confidence():
    trades = ([_trade(f"H{i}", "LONG", 0.05, confidence="HIGH") for i in range(5)]
              + [_trade(f"L{i}", "LONG", -0.05, confidence="LOW") for i in range(5)])
    predicted = [0.05] * 10
    report = compute_strategy_outcome_report(trades, n_decisions_considered=10, predicted_impacts=predicted,
                                              transaction_cost=0.0)
    assert report.hit_rate_by_confidence["HIGH"] == 1.0
    assert report.hit_rate_by_confidence["LOW"] == 0.0


def test_report_mean_mfe_and_mae_are_aggregated():
    trades = [_trade("A", "LONG", 0.05, mfe=0.10, mae=-0.02), _trade("B", "LONG", 0.03, mfe=0.06, mae=-0.04)]
    report = compute_strategy_outcome_report(trades, n_decisions_considered=2, predicted_impacts=[0.05, 0.03],
                                              transaction_cost=0.0)
    assert abs(report.mean_mfe - 0.08) < 1e-9
    assert abs(report.mean_mae - (-0.03)) < 1e-9
