from market_agent.experiment.portfolio_metrics import TradeRecord, compute_portfolio_metrics


def _trade(day, predicted_impact, realized, horizon=20):
    return TradeRecord(entity="ACME", triggered_at=f"2024-01-{day:02d}T00:00:00+00:00",
                        horizon_days=horizon, predicted_impact=predicted_impact, realized_abnormal_return=realized)


def test_too_few_trades_returns_none_metrics():
    report = compute_portfolio_metrics([_trade(1, 0.02, 0.01)])
    assert report.n_trades == 1
    assert report.expected_value is None
    assert "Only 1 trade" in report.evidence[0]


def test_correct_direction_trades_produce_positive_expected_value():
    trades = [_trade(1, 0.02, 0.03), _trade(2, 0.02, 0.025), _trade(3, -0.02, -0.03), _trade(4, -0.02, -0.02)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.n_trades == 4
    assert report.expected_value > 0
    assert report.expected_value == report.expected_value_after_costs  # zero cost in this call


def test_transaction_costs_reduce_expected_value():
    trades = [_trade(1, 0.02, 0.03), _trade(2, 0.02, 0.025)]
    no_cost = compute_portfolio_metrics(trades, transaction_cost=0.0)
    with_cost = compute_portfolio_metrics(trades, transaction_cost=0.01)
    assert with_cost.expected_value_after_costs < no_cost.expected_value_after_costs
    assert abs((no_cost.expected_value_after_costs - with_cost.expected_value_after_costs) - 0.01) < 1e-9


def test_wrong_direction_trades_produce_negative_expected_value():
    trades = [_trade(1, 0.02, -0.03), _trade(2, -0.02, 0.03), _trade(3, 0.02, -0.01)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.expected_value < 0


def test_max_drawdown_reflects_a_losing_streak_after_gains():
    # gains first, then a losing streak that pulls the equity curve below its running peak
    trades = [_trade(1, 0.02, 0.05), _trade(2, 0.02, 0.05),
              _trade(3, 0.02, -0.06), _trade(4, 0.02, -0.06), _trade(5, 0.02, -0.06)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.max_drawdown < 0
    # peak was 0.10 after trade 2; equity after trade 5 is 0.10 - 0.18 = -0.08 -> drawdown -0.18
    assert abs(report.max_drawdown - (-0.18)) < 1e-9


def test_no_drawdown_when_equity_curve_is_monotonically_rising():
    trades = [_trade(1, 0.02, 0.01), _trade(2, 0.02, 0.01), _trade(3, 0.02, 0.01)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.max_drawdown == 0.0


def test_sharpe_is_none_when_returns_have_no_variance():
    trades = [_trade(1, 0.02, 0.02), _trade(2, 0.02, 0.02), _trade(3, 0.02, 0.02)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.sharpe_per_trade is None  # zero stdev - undefined, not fabricated


def test_sharpe_annualized_scales_with_trade_frequency():
    trades = [_trade(d, 0.02, 0.03 if d % 2 else 0.01) for d in range(1, 11)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.sharpe_per_trade is not None
    assert report.sharpe_annualized is not None
    assert abs(report.sharpe_annualized) > abs(report.sharpe_per_trade)  # annualization scales it up


def test_turnover_reflects_trade_frequency_over_the_window():
    trades = [_trade(d, 0.02, 0.03) for d in range(1, 11)]  # 10 trades over a 9-day window
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.turnover_trades_per_year > 0
    assert report.evaluation_window_days == 9.0


def test_short_trade_direction_is_correctly_signed():
    trades = [_trade(1, -0.02, -0.05), _trade(2, -0.02, -0.04)]  # predicted down, realized down = a WIN for shorts
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.expected_value > 0


def test_trades_are_ordered_chronologically_regardless_of_input_order():
    trades = [_trade(5, 0.02, 0.01), _trade(1, 0.02, 0.01), _trade(3, 0.02, 0.01)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.evaluation_window_days == 4.0  # Jan 1 to Jan 5, regardless of input order


def test_sortino_is_none_with_fewer_than_two_losing_trades():
    trades = [_trade(1, 0.02, 0.03), _trade(2, 0.02, 0.01), _trade(3, 0.02, 0.02)]  # zero losers
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.sortino_per_trade is None  # not 0.0 - "no downside data" != "measured zero risk"


def test_sortino_exceeds_sharpe_when_variance_is_mostly_upside():
    """The whole point of Sortino: large UPSIDE variance must not penalize
    the ratio the way Sharpe's total-variance denominator does. Two small,
    consistent losses next to occasionally huge wins should score better
    under Sortino than under Sharpe."""
    trades = [_trade(1, 0.02, 0.01), _trade(2, 0.02, 0.30), _trade(3, 0.02, -0.01),
              _trade(4, 0.02, 0.25), _trade(5, 0.02, -0.015), _trade(6, 0.02, 0.28)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.sortino_per_trade is not None and report.sharpe_per_trade is not None
    assert report.sortino_per_trade > report.sharpe_per_trade


def test_sortino_annualized_scales_with_trade_frequency():
    trades = [_trade(d, 0.02, 0.03 if d % 2 else -0.01) for d in range(1, 11)]
    report = compute_portfolio_metrics(trades, transaction_cost=0.0)
    assert report.sortino_per_trade is not None
    assert report.sortino_annualized is not None
    assert abs(report.sortino_annualized) > abs(report.sortino_per_trade)


def test_too_few_trades_leaves_sortino_none_too():
    report = compute_portfolio_metrics([_trade(1, 0.02, 0.01)])
    assert report.sortino_per_trade is None
    assert report.sortino_annualized is None
