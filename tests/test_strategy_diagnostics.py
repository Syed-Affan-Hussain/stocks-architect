from datetime import datetime, timedelta, timezone

from market_agent.outcomes.ohlcv import Bar, OHLCVProvider
from market_agent.strategy.outcome_engine import TradeOutcome, compute_trade_outcome
from market_agent.strategy.strategy_diagnostics import (
    bootstrap_confidence_interval, holding_period_sensitivity, regime_stability, run_placebo_strategy_test,
    run_strategy_permutation_test, run_strategy_temporal_stability, segment_degradation,
    threshold_sensitivity, transaction_cost_sensitivity,
)

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


def _trade(entity, realized_return, entry_offset_days=0, regime="RISK_ON", path_returns=None):
    return TradeOutcome(entity=entity, action="LONG", entry_date=(START + timedelta(days=entry_offset_days)).isoformat(),
                         entry_price=100, exit_date=(START + timedelta(days=entry_offset_days + 20)).isoformat(),
                         exit_price=100 * (1 + realized_return), horizon_days=20, holding_days=20,
                         raw_entity_return=realized_return, realized_return=realized_return, mfe=0.05, mae=-0.02,
                         hit_invalidation=False, invalidation_day=None, confidence="HIGH", regime=regime,
                         path_returns=path_returns or [])


def test_bootstrap_ci_too_few_values():
    assert bootstrap_confidence_interval([0.05]) is None


def test_bootstrap_ci_brackets_the_true_mean_for_tight_data():
    values = [0.05, 0.051, 0.049, 0.052, 0.048] * 5
    ci = bootstrap_confidence_interval(values)
    assert ci is not None
    low, high = ci
    assert low <= 0.05 <= high
    assert high - low < 0.02  # tight data -> a tight CI


def test_transaction_cost_sensitivity_reduces_expectancy_as_cost_grows():
    trades = [_trade(f"E{i}", 0.05) for i in range(10)]
    predicted = [0.05] * 10
    results = transaction_cost_sensitivity(trades, predicted, cost_grid=(0.0, 0.01))
    assert results[0.0].expectancy > results[0.01].expectancy


def test_threshold_sensitivity_uses_the_callback_at_each_grid_point():
    class FakeDecision:
        def __init__(self, action):
            self.action = action

    def decision_fn(multiple):
        # simulate: fewer trades qualify as the threshold grows
        n_long = max(10 - int(multiple), 0)
        return [FakeDecision("LONG") for _ in range(n_long)] + [FakeDecision("ABSTAIN") for _ in range(20 - n_long)]

    results = threshold_sensitivity(decision_fn, cost_margin_grid=(1.0, 5.0, 9.0))
    assert results[1.0]["LONG"] == 9
    assert results[9.0]["LONG"] == 1
    assert results[1.0]["LONG"] > results[9.0]["LONG"]


def test_holding_period_sensitivity_reads_real_path_returns():
    trades = [_trade("A", 0.10, path_returns=[0.0, 0.02, 0.05, 0.08, 0.10]),
              _trade("B", 0.06, path_returns=[0.0, 0.01, 0.03, 0.05, 0.06])]
    results = holding_period_sensitivity(trades, day_grid=(1, 4, 10))
    assert abs(results[1] - 0.015) < 1e-9   # mean of 0.02 and 0.01
    assert abs(results[4] - 0.08) < 1e-9    # mean of 0.10 and 0.06
    assert results[10] is None              # no trade's path reaches day 10


def test_regime_stability_splits_by_entry_regime():
    trades = [_trade(f"ON{i}", 0.05, regime="RISK_ON") for i in range(5)] + \
             [_trade(f"OFF{i}", -0.05, regime="RISK_OFF") for i in range(5)]
    predicted = [0.05] * 10
    results = regime_stability(trades, predicted)
    assert set(results.keys()) == {"RISK_ON", "RISK_OFF"}
    assert results["RISK_ON"].expectancy > 0
    assert results["RISK_OFF"].expectancy < 0


def test_segment_degradation_reports_each_segment_independently():
    trades_by_segment = {"VALIDATE": [_trade(f"V{i}", 0.05) for i in range(10)],
                          "TEST": [_trade(f"T{i}", -0.01) for i in range(10)]}
    predicted = {"VALIDATE": [0.05] * 10, "TEST": [0.05] * 10}
    results = segment_degradation(trades_by_segment, predicted)
    assert results["VALIDATE"].expectancy > results["TEST"].expectancy


def test_placebo_strategy_test_produces_real_trades_from_random_entries():
    ohlcv = FakeOHLCV()
    bars = [_bar(i, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(200)]
    ohlcv.set_bars("ACME", bars)
    report = run_placebo_strategy_test(ohlcv, ["ACME"], START, START + timedelta(days=150),
                                        n_placebo_trades=10, horizon_days=10)
    assert report.n_trades > 0


def test_strategy_permutation_insufficient_n():
    observed = [_trade(f"E{i}", 0.05) for i in range(5)]
    pool = [_trade(f"P{i}", 0.0) for i in range(5)]
    result = run_strategy_permutation_test(observed, pool)
    assert result.status == "INSUFFICIENT_N"


def test_strategy_permutation_survives_when_observed_is_a_real_outlier():
    pool = [_trade(f"P{i}", 0.001 if i % 2 == 0 else -0.001) for i in range(200)]
    observed = [_trade(f"O{i}", 0.15 + (0.005 if i % 2 == 0 else -0.005)) for i in range(20)]
    result = run_strategy_permutation_test(observed, pool + observed, n_permutations=300)
    assert result.status == "SURVIVES_PERMUTATION"


def test_strategy_temporal_stability_detects_a_sign_flip():
    early = [_trade(f"E{i}", 0.08, entry_offset_days=i) for i in range(15)]
    late = [_trade(f"L{i}", -0.08, entry_offset_days=500 + i) for i in range(15)]
    result = run_strategy_temporal_stability(early + late)
    assert result.status == "UNSTABLE_ACROSS_TIME"
