from datetime import datetime, timedelta, timezone

from market_agent.concepts.technical_context import build_technical_context
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data: dict[str, list[Bar]] = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff_start = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff_start]


def _bar(day_offset, o, h, l, c, v=1_000_000):
    return Bar(date=START + timedelta(days=day_offset), open=o, high=h, low=l, close=c, volume=v)


def _trend_series(n, start_price=100.0, daily_pct=0.005, start_offset=0, volume=1_000_000):
    bars = []
    price = start_price
    for i in range(n):
        o = price
        price = price * (1 + daily_pct)
        c = price
        h, l = max(o, c) * 1.001, min(o, c) * 0.999
        bars.append(_bar(start_offset + i, o, h, l, c, volume))
    return bars


def test_insufficient_history_leaves_everything_unknown():
    ohlcv = FakeOHLCV()
    ohlcv.set_bars("ACME", [_bar(0, 100, 101, 99, 100.5)])
    ctx = build_technical_context(ohlcv, "ACME", START + timedelta(days=1))
    assert ctx.trend_direction == "UNKNOWN"
    assert ctx.momentum_state == "UNKNOWN"
    assert ctx.breakout_state == "UNKNOWN"


def test_uptrend_detected_from_rising_closes():
    ohlcv = FakeOHLCV()
    bars = _trend_series(40, start_price=100.0, daily_pct=0.01)
    ohlcv.set_bars("ACME", bars)
    as_of = bars[-1].date
    ctx = build_technical_context(ohlcv, "ACME", as_of)
    assert ctx.trend_direction == "UP"
    assert ctx.momentum_state == "POSITIVE"
    assert ctx.momentum_roc_20d > 0


def test_downtrend_detected_from_falling_closes():
    ohlcv = FakeOHLCV()
    bars = _trend_series(40, start_price=100.0, daily_pct=-0.01)
    ohlcv.set_bars("ACME", bars)
    as_of = bars[-1].date
    ctx = build_technical_context(ohlcv, "ACME", as_of)
    assert ctx.trend_direction == "DOWN"
    assert ctx.momentum_state == "NEGATIVE"


def test_breakout_up_when_close_clears_rolling_high():
    ohlcv = FakeOHLCV()
    bars = _trend_series(25, start_price=100.0, daily_pct=0.0)  # flat base, low volatility range
    spike = _bar(25, 100.0, 130.0, 99.0, 128.0)  # far above the flat 20d range
    ohlcv.set_bars("ACME", bars + [spike])
    ctx = build_technical_context(ohlcv, "ACME", spike.date)
    assert ctx.breakout_state == "BREAKOUT_UP"


def test_mean_reversion_overextended_high_on_extreme_spike():
    ohlcv = FakeOHLCV()
    bars = _trend_series(25, start_price=100.0, daily_pct=0.0005)  # tight, low-variance base
    spike = _bar(25, 100.0, 200.0, 100.0, 195.0)  # extreme outlier vs a tight recent range
    ohlcv.set_bars("ACME", bars + [spike])
    ctx = build_technical_context(ohlcv, "ACME", spike.date)
    assert ctx.mean_reversion_state == "OVEREXTENDED_HIGH"
    assert ctx.mean_reversion_zscore > 2.0


def test_price_action_inside_bar():
    ohlcv = FakeOHLCV()
    prev = _bar(0, 100, 110, 90, 105)
    cur = _bar(1, 103, 108, 95, 104)  # fully inside prev's high/low range
    ohlcv.set_bars("ACME", [prev, cur])
    ctx = build_technical_context(ohlcv, "ACME", cur.date)
    assert ctx.price_action_pattern == "INSIDE_BAR"


def test_price_action_outside_bar():
    ohlcv = FakeOHLCV()
    prev = _bar(0, 100, 105, 98, 102)
    cur = _bar(1, 101, 110, 90, 108)  # engulfs prev's high/low range
    ohlcv.set_bars("ACME", [prev, cur])
    ctx = build_technical_context(ohlcv, "ACME", cur.date)
    assert ctx.price_action_pattern == "OUTSIDE_BAR"


def test_price_action_bullish_engulfing():
    ohlcv = FakeOHLCV()
    prev = _bar(0, 105, 106, 98, 99)          # red bar: close < open
    # Real body engulfs prev's real body (close >= prev.open, open <= prev.close) WITHOUT also being
    # an outside bar (high stays below prev.high, low dips below prev.low on only one side) - an
    # engulfing bar that also happens to be an outside bar is classified OUTSIDE_BAR first (see
    # technical_context.py's priority order), so this fixture deliberately isolates the engulfing case.
    cur = _bar(1, 98, 105.8, 97.5, 105.5)
    ohlcv.set_bars("ACME", [prev, cur])
    ctx = build_technical_context(ohlcv, "ACME", cur.date)
    assert ctx.price_action_pattern == "BULLISH_ENGULFING"


def test_gap_up_detected():
    ohlcv = FakeOHLCV()
    prev = _bar(0, 100, 101, 99, 100)
    cur = _bar(1, 106, 108, 105, 107)  # opens 6% above prior close
    ohlcv.set_bars("ACME", [prev, cur])
    ctx = build_technical_context(ohlcv, "ACME", cur.date)
    assert ctx.gap_state == "GAP_UP"
    assert ctx.gap_pct > 0.02


def test_relative_volume_high_on_volume_spike():
    ohlcv = FakeOHLCV()
    bars = _trend_series(25, start_price=100.0, daily_pct=0.0, volume=1_000_000)
    spike = _bar(25, 100, 101, 99, 100.5, v=5_000_000)
    ohlcv.set_bars("ACME", bars + [spike])
    ctx = build_technical_context(ohlcv, "ACME", spike.date)
    assert ctx.relative_volume_state == "HIGH_RVOL"
    assert ctx.relative_volume_raw > 1.5


def test_market_structure_higher_highs_higher_lows():
    ohlcv = FakeOHLCV()
    bars = _trend_series(45, start_price=100.0, daily_pct=0.01)
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.market_structure == "HIGHER_HIGHS_HIGHER_LOWS"


def test_moving_average_bullish_stack_needs_two_hundred_bars():
    ohlcv = FakeOHLCV()
    short_bars = _trend_series(60, start_price=100.0, daily_pct=0.005)
    ohlcv.set_bars("ACME", short_bars)
    ctx_short = build_technical_context(ohlcv, "ACME", short_bars[-1].date)
    assert ctx_short.ma_stack == "UNKNOWN"  # honestly insufficient history, not guessed

    long_bars = _trend_series(220, start_price=50.0, daily_pct=0.004)
    ohlcv.set_bars("ACME", long_bars)
    ctx_long = build_technical_context(ohlcv, "ACME", long_bars[-1].date)
    assert ctx_long.ma_stack == "BULLISH_STACK"


def test_relative_strength_outperforming_benchmark():
    ohlcv = FakeOHLCV()
    entity_bars = _trend_series(25, start_price=100.0, daily_pct=0.02)   # strong uptrend
    bench_bars = _trend_series(25, start_price=400.0, daily_pct=0.001)   # flat benchmark
    ohlcv.set_bars("ACME", entity_bars)
    ohlcv.set_bars("SPY", bench_bars)
    ctx = build_technical_context(ohlcv, "ACME", entity_bars[-1].date, benchmark_ticker="SPY")
    assert ctx.relative_strength_state == "OUTPERFORMING"
    assert ctx.relative_strength_20d > 0


def test_vwap_state_reflects_close_vs_volume_weighted_proxy():
    ohlcv = FakeOHLCV()
    bars = _trend_series(25, start_price=100.0, daily_pct=0.0, volume=1_000_000)
    spike = _bar(25, 100, 140, 99, 138)  # closes well above the recent volume-weighted price level
    ohlcv.set_bars("ACME", bars + [spike])
    ctx = build_technical_context(ohlcv, "ACME", spike.date)
    assert ctx.vwap_state == "ABOVE_VWAP_PROXY"
    assert ctx.vwap_proxy_20d is not None


def test_pullback_in_uptrend_flagged_on_shallow_retracement():
    ohlcv = FakeOHLCV()
    up_bars = _trend_series(30, start_price=100.0, daily_pct=0.015)
    peak_close = up_bars[-1].close
    retrace_close = peak_close * 0.96  # ~4% pullback - inside the disclosed 2-10% band
    retrace = _bar(30, retrace_close * 1.01, retrace_close * 1.01, retrace_close * 0.99, retrace_close)
    ohlcv.set_bars("ACME", up_bars + [retrace])
    ctx = build_technical_context(ohlcv, "ACME", retrace.date)
    assert ctx.trend_direction == "UP"
    assert ctx.pullback_state == "PULLBACK_IN_UPTREND"


# --- stage 7 item 7: close-location value, moving-average slope, liquidity regime ---

def test_clv_upper_range_when_closes_consistently_sit_near_the_bar_high():
    ohlcv = FakeOHLCV()
    bars = [_bar(i, 100, 105, 95, 104) for i in range(12)]  # CLV = ((104-95)-(105-104))/10 = 0.8 each bar
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.clv_state == "UPPER_RANGE"
    assert ctx.clv_raw > 0.3


def test_clv_lower_range_when_closes_consistently_sit_near_the_bar_low():
    ohlcv = FakeOHLCV()
    bars = [_bar(i, 100, 105, 95, 96) for i in range(12)]  # CLV = ((96-95)-(105-96))/10 = -0.8 each bar
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.clv_state == "LOWER_RANGE"
    assert ctx.clv_raw < -0.3


def test_ma_slope_rising_in_an_uptrend():
    ohlcv = FakeOHLCV()
    bars = _trend_series(40, start_price=100.0, daily_pct=0.01)
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.ma_slope_state == "RISING"
    assert ctx.ma_slope_pct > 0


def test_ma_slope_falling_in_a_downtrend():
    ohlcv = FakeOHLCV()
    bars = _trend_series(40, start_price=100.0, daily_pct=-0.01)
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.ma_slope_state == "FALLING"
    assert ctx.ma_slope_pct < 0


def test_liquidity_regime_high_when_recent_dollar_volume_outpaces_the_longer_baseline():
    ohlcv = FakeOHLCV()
    baseline = [_bar(i, 100, 101, 99, 100, v=1_000_000) for i in range(40)]
    recent_high_volume = [_bar(40 + i, 100, 101, 99, 100, v=5_000_000) for i in range(20)]
    ohlcv.set_bars("ACME", baseline + recent_high_volume)
    ctx = build_technical_context(ohlcv, "ACME", recent_high_volume[-1].date)
    assert ctx.liquidity_regime == "HIGH_LIQUIDITY"
    assert ctx.liquidity_dollar_volume_ratio > 1.3


def test_liquidity_regime_low_when_recent_dollar_volume_dries_up():
    ohlcv = FakeOHLCV()
    baseline = [_bar(i, 100, 101, 99, 100, v=5_000_000) for i in range(40)]
    recent_low_volume = [_bar(40 + i, 100, 101, 99, 100, v=1_000_000) for i in range(20)]
    ohlcv.set_bars("ACME", baseline + recent_low_volume)
    ctx = build_technical_context(ohlcv, "ACME", recent_low_volume[-1].date)
    assert ctx.liquidity_regime == "LOW_LIQUIDITY"
    assert ctx.liquidity_dollar_volume_ratio < 0.7


def test_liquidity_regime_unknown_below_the_long_window():
    ohlcv = FakeOHLCV()
    bars = [_bar(i, 100, 101, 99, 100, v=1_000_000) for i in range(30)]  # below LIQUIDITY_LONG_WINDOW=60
    ohlcv.set_bars("ACME", bars)
    ctx = build_technical_context(ohlcv, "ACME", bars[-1].date)
    assert ctx.liquidity_regime == "UNKNOWN"
    assert ctx.liquidity_dollar_volume_ratio is None


def test_clv_ma_slope_liquidity_are_in_the_shared_dimension_registries():
    from market_agent.concepts.ontology import TradingConcept
    from market_agent.concepts.technical_context import DIMENSION_TO_CONCEPT, TECHNICAL_STATE_FIELD_NAMES

    for field in ("clv_state", "ma_slope_state", "liquidity_regime"):
        assert field in TECHNICAL_STATE_FIELD_NAMES
        assert field in DIMENSION_TO_CONCEPT
    assert DIMENSION_TO_CONCEPT["clv_state"] == TradingConcept.CLOSE_LOCATION_VALUE
    assert DIMENSION_TO_CONCEPT["ma_slope_state"] == TradingConcept.TREND  # not a new concept - see ontology.py
    assert DIMENSION_TO_CONCEPT["liquidity_regime"] == TradingConcept.LIQUIDITY_REGIME


def test_to_dict_includes_all_fields():
    ohlcv = FakeOHLCV()
    ctx = build_technical_context(ohlcv, "ACME", START)
    d = ctx.to_dict()
    assert d["entity"] == "ACME"
    assert "trend_direction" in d and "volatility_state" in d and "ma_stack" in d
    assert "clv_state" in d and "ma_slope_state" in d and "liquidity_regime" in d
