"""Point-in-time TechnicalMarketContext - stage 6's real, computed
representation of 19 of the 22 canonical Trading Concepts (ontology.py),
built ONLY from bars strictly on-or-before `as_of` (outcomes/ohlcv.py's
OHLCVProvider.bars() already enforces that cutoff; nothing here reaches
past it). OPENING_RANGE, SECTOR_CONTEXT, and RISK_MANAGEMENT are not
represented here - see ontology.py's module docstring for why each is a
disclosed gap rather than an approximation, along with the further,
explicitly-named gaps (market breadth, cross-security correlation) that
were never concepts at all.

STAGE 7 ITEM 7 added three new fields: `clv_state` (close-location value -
where the close sits within its own bar's range, a NEW concept),
`ma_slope_state` (moving-average slope - a more granular measurement of
the SAME TREND concept `trend_direction` already names, not a new
concept), and `liquidity_regime` (an entity's own sustained dollar-volume
level relative to its own longer-run baseline - a NEW concept, and
deliberately NOT a cross-security liquidity comparison; see ontology.py).

EVERY THRESHOLD BELOW IS FIXED AND DISCLOSED, chosen once from standard
technical-analysis convention (not fit to any backtest result) - same
discipline as retrieval/regime.py's RISK_OFF_THRESHOLD/RISK_ON_THRESHOLD
and retrieval/similarity.py's bucket boundaries. Changing one later is a
disclosed, versioned decision, never a silent retune after seeing a
hypothesis-testing result.

Each qualitative field below is a discrete, categorical label (never a
raw float) specifically because learn/hypothesis.py's conditioning
dimensions and the relationship-matching machinery (agents/adaptive_agent.py's
_condition_matches_context) work by exact-value dict matching, the same
way `regime`/`prior_return_bucket`/`vol_bucket` already do - a float here
would never match anything. Raw diagnostic floats are ALSO kept (suffixed
`_raw` or self-descriptive) for dashboard/audit display, but are never
themselves used as a matching key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from market_agent.concepts.ontology import TradingConcept
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider

# --- disclosed, fixed window/threshold constants ---
TREND_MA_PERIOD = 20
TREND_SLOPE_LOOKBACK = 10
MOMENTUM_LOOKBACK = 20
MOMENTUM_THRESHOLD = 0.02
MEAN_REVERSION_LOOKBACK = 20
MEAN_REVERSION_Z_THRESHOLD = 2.0
BREAKOUT_LOOKBACK = 20
PULLBACK_MIN_PCT = 0.02
PULLBACK_MAX_PCT = 0.10
SUPPORT_RESISTANCE_LOOKBACK = 20
SUPPORT_RESISTANCE_PROXIMITY = 0.02
VOLATILITY_SHORT_WINDOW = 5
VOLATILITY_LONG_WINDOW = 20
VOLATILITY_COMPRESSION_RATIO = 0.7
VOLATILITY_EXPANSION_RATIO = 1.3
RELATIVE_VOLUME_LOOKBACK = 20
RELATIVE_VOLUME_HIGH = 1.5
RELATIVE_VOLUME_LOW = 0.5
VWAP_PROXY_LOOKBACK = 20
VWAP_PROXY_THRESHOLD = 0.005
MA_STACK_PERIODS = (20, 50, 200)
MARKET_STRUCTURE_HALF_WINDOW = 20
GAP_THRESHOLD = 0.02
RELATIVE_STRENGTH_LOOKBACK = 20
RELATIVE_STRENGTH_THRESHOLD = 0.02
CLOSE_LOCATION_VALUE_LOOKBACK = 10
CLV_UPPER_THRESHOLD = 0.3
CLV_LOWER_THRESHOLD = -0.3
MA_SLOPE_FLAT_THRESHOLD = 0.01  # % change in the trend MA over TREND_SLOPE_LOOKBACK days
LIQUIDITY_SHORT_WINDOW = 20
LIQUIDITY_LONG_WINDOW = 60
LIQUIDITY_HIGH_RATIO = 1.3
LIQUIDITY_LOW_RATIO = 0.7
MAX_LOOKBACK_CALENDAR_DAYS = 320  # enough calendar days to cover a 200-trading-day window + holidays


@dataclass
class TechnicalMarketContext:
    entity: str
    as_of: str

    trend_direction: str = "UNKNOWN"           # "UP" | "DOWN" | "FLAT" | "UNKNOWN"
    trend_ma20: float | None = None

    momentum_state: str = "UNKNOWN"            # "POSITIVE" | "NEGATIVE" | "FLAT" | "UNKNOWN"
    momentum_roc_20d: float | None = None

    mean_reversion_state: str = "UNKNOWN"      # "OVEREXTENDED_HIGH" | "OVEREXTENDED_LOW" | "NORMAL" | "UNKNOWN"
    mean_reversion_zscore: float | None = None

    breakout_state: str = "UNKNOWN"            # "BREAKOUT_UP" | "BREAKOUT_DOWN" | "NONE" | "UNKNOWN"
    distance_from_rolling_high_pct: float | None = None
    distance_from_rolling_low_pct: float | None = None

    pullback_state: str = "UNKNOWN"            # "PULLBACK_IN_UPTREND" | "PULLBACK_IN_DOWNTREND" | "NONE" | "UNKNOWN"

    price_action_pattern: str = "UNKNOWN"      # "INSIDE_BAR"|"OUTSIDE_BAR"|"BULLISH_ENGULFING"|"BEARISH_ENGULFING"|"NONE"|"UNKNOWN"

    support_resistance_state: str = "UNKNOWN"  # "NEAR_RESISTANCE" | "NEAR_SUPPORT" | "MID_RANGE" | "UNKNOWN"

    volatility_state: str = "UNKNOWN"          # "COMPRESSION" | "EXPANSION" | "NORMAL" | "UNKNOWN"
    atr_ratio: float | None = None

    latest_volume: float | None = None

    relative_volume_state: str = "UNKNOWN"     # "HIGH_RVOL" | "LOW_RVOL" | "NORMAL_RVOL" | "UNKNOWN"
    relative_volume_raw: float | None = None

    vwap_state: str = "UNKNOWN"                # "ABOVE_VWAP_PROXY" | "BELOW_VWAP_PROXY" | "AT_VWAP_PROXY" | "UNKNOWN"
    vwap_proxy_20d: float | None = None

    ma_stack: str = "UNKNOWN"                  # "BULLISH_STACK" | "BEARISH_STACK" | "MIXED_STACK" | "UNKNOWN"

    market_structure: str = "UNKNOWN"          # "HIGHER_HIGHS_HIGHER_LOWS" | "LOWER_HIGHS_LOWER_LOWS" | "MIXED" | "UNKNOWN"

    gap_state: str = "UNKNOWN"                 # "GAP_UP" | "GAP_DOWN" | "NO_GAP" | "UNKNOWN"
    gap_pct: float | None = None

    relative_strength_state: str = "UNKNOWN"   # "OUTPERFORMING" | "UNDERPERFORMING" | "IN_LINE" | "UNKNOWN"
    relative_strength_20d: float | None = None

    mtf_confirmation: str = "UNKNOWN"          # "CONFIRMED" | "CONFLICTED" | "UNKNOWN"

    clv_state: str = "UNKNOWN"                 # "UPPER_RANGE" | "LOWER_RANGE" | "MID_RANGE" | "UNKNOWN"
    clv_raw: float | None = None

    ma_slope_state: str = "UNKNOWN"            # "RISING" | "FALLING" | "FLAT" | "UNKNOWN"
    ma_slope_pct: float | None = None

    liquidity_regime: str = "UNKNOWN"          # "HIGH_LIQUIDITY" | "LOW_LIQUIDITY" | "NORMAL_LIQUIDITY" | "UNKNOWN"
    liquidity_dollar_volume_ratio: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _true_range(bar: Bar, prev_close: float | None) -> float:
    if prev_close is None:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))


def _avg_true_range(bars: list[Bar], window: int) -> float | None:
    if len(bars) < window + 1:
        return None
    recent = bars[-window:]
    trs = []
    for i, bar in enumerate(recent):
        idx = len(bars) - window + i
        prev_close = bars[idx - 1].close if idx > 0 else None
        trs.append(_true_range(bar, prev_close))
    return sum(trs) / len(trs)


def _ma_slope(closes: list[float]) -> tuple[float, float] | None:
    """(raw_slope, ma_series_start) of the TREND_MA_PERIOD SMA over
    TREND_SLOPE_LOOKBACK days, or None if there isn't enough history.
    Shared by `_trend_from_closes` (sign only) and stage 7's
    `ma_slope_state`/`ma_slope_pct` (magnitude, normalized) so the two can
    never silently compute different slopes for the same field."""
    ma = _sma(closes, TREND_MA_PERIOD)
    if ma is None or len(closes) < TREND_MA_PERIOD + TREND_SLOPE_LOOKBACK:
        return None
    ma_series = [_sma(closes[:i + 1], TREND_MA_PERIOD) for i in
                 range(len(closes) - TREND_SLOPE_LOOKBACK - 1, len(closes))]
    ma_series = [m for m in ma_series if m is not None]
    if len(ma_series) < 2:
        return None
    return ma_series[-1] - ma_series[0], ma_series[0]


def _trend_from_closes(closes: list[float]) -> str:
    ma = _sma(closes, TREND_MA_PERIOD)
    slope_info = _ma_slope(closes)
    if ma is None or slope_info is None:
        return "UNKNOWN"
    slope, _ = slope_info
    latest_close = closes[-1]
    if latest_close > ma and slope > 0:
        return "UP"
    if latest_close < ma and slope < 0:
        return "DOWN"
    return "FLAT"


def build_technical_context(ohlcv: OHLCVProvider, entity: str, as_of: datetime,
                             benchmark_ticker: str = "SPY") -> TechnicalMarketContext:
    ctx = TechnicalMarketContext(entity=entity, as_of=as_of.isoformat())
    bars = ohlcv.bars(entity, as_of, lookback_days=MAX_LOOKBACK_CALENDAR_DAYS)
    if len(bars) < 2:
        return ctx  # everything stays UNKNOWN/None - honestly insufficient history, not guessed at

    closes = [b.close for b in bars]
    latest = bars[-1]

    # TREND
    ctx.trend_direction = _trend_from_closes(closes)
    ctx.trend_ma20 = _sma(closes, TREND_MA_PERIOD)

    # MOVING-AVERAGE SLOPE (stage 7 item 7) - same underlying slope _trend_from_closes already uses
    # for its sign, exposed here as its own normalized, discrete field (see ontology.py for why this
    # maps back onto TREND rather than becoming a new concept).
    slope_info = _ma_slope(closes)
    if slope_info is not None:
        slope, ma_series_start = slope_info
        if ma_series_start:
            slope_pct = slope / ma_series_start
            ctx.ma_slope_pct = slope_pct
            ctx.ma_slope_state = ("RISING" if slope_pct > MA_SLOPE_FLAT_THRESHOLD else
                                   "FALLING" if slope_pct < -MA_SLOPE_FLAT_THRESHOLD else "FLAT")

    # MOMENTUM
    if len(closes) > MOMENTUM_LOOKBACK:
        base = closes[-MOMENTUM_LOOKBACK - 1]
        if base:
            roc = (closes[-1] - base) / base
            ctx.momentum_roc_20d = roc
            ctx.momentum_state = ("POSITIVE" if roc > MOMENTUM_THRESHOLD else
                                   "NEGATIVE" if roc < -MOMENTUM_THRESHOLD else "FLAT")

    # MEAN_REVERSION
    if len(closes) >= MEAN_REVERSION_LOOKBACK:
        window = closes[-MEAN_REVERSION_LOOKBACK:]
        mean = sum(window) / len(window)
        sd = _stdev(window)
        if sd and sd > 0:
            z = (closes[-1] - mean) / sd
            ctx.mean_reversion_zscore = z
            ctx.mean_reversion_state = ("OVEREXTENDED_HIGH" if z > MEAN_REVERSION_Z_THRESHOLD else
                                         "OVEREXTENDED_LOW" if z < -MEAN_REVERSION_Z_THRESHOLD else "NORMAL")

    # BREAKOUT / SUPPORT_RESISTANCE (share the same rolling-window high/low, excluding the latest bar)
    if len(bars) > BREAKOUT_LOOKBACK:
        window_bars = bars[-BREAKOUT_LOOKBACK - 1:-1]
        rolling_high = max(b.high for b in window_bars)
        rolling_low = min(b.low for b in window_bars)
        if rolling_high > 0:
            ctx.distance_from_rolling_high_pct = (rolling_high - latest.close) / rolling_high
        if rolling_low > 0:
            ctx.distance_from_rolling_low_pct = (latest.close - rolling_low) / rolling_low

        if latest.close > rolling_high:
            ctx.breakout_state = "BREAKOUT_UP"
        elif latest.close < rolling_low:
            ctx.breakout_state = "BREAKOUT_DOWN"
        else:
            ctx.breakout_state = "NONE"

        if ctx.distance_from_rolling_high_pct is not None and ctx.distance_from_rolling_high_pct <= SUPPORT_RESISTANCE_PROXIMITY:
            ctx.support_resistance_state = "NEAR_RESISTANCE"
        elif ctx.distance_from_rolling_low_pct is not None and ctx.distance_from_rolling_low_pct <= SUPPORT_RESISTANCE_PROXIMITY:
            ctx.support_resistance_state = "NEAR_SUPPORT"
        else:
            ctx.support_resistance_state = "MID_RANGE"

        # PULLBACK - a retracement AGAINST the prevailing trend, so it's only defined once TREND is known
        if ctx.trend_direction == "UP" and ctx.distance_from_rolling_high_pct is not None:
            ctx.pullback_state = ("PULLBACK_IN_UPTREND"
                                   if PULLBACK_MIN_PCT <= ctx.distance_from_rolling_high_pct <= PULLBACK_MAX_PCT
                                   else "NONE")
        elif ctx.trend_direction == "DOWN" and ctx.distance_from_rolling_low_pct is not None:
            ctx.pullback_state = ("PULLBACK_IN_DOWNTREND"
                                   if PULLBACK_MIN_PCT <= ctx.distance_from_rolling_low_pct <= PULLBACK_MAX_PCT
                                   else "NONE")
        elif ctx.trend_direction in ("UP", "DOWN", "FLAT"):
            ctx.pullback_state = "NONE"

    # PRICE_ACTION (needs the latest 2 bars)
    if len(bars) >= 2:
        prev, cur = bars[-2], bars[-1]
        if cur.high <= prev.high and cur.low >= prev.low:
            ctx.price_action_pattern = "INSIDE_BAR"
        elif cur.high >= prev.high and cur.low <= prev.low:
            ctx.price_action_pattern = "OUTSIDE_BAR"
        elif cur.close > cur.open and prev.close < prev.open and cur.close >= prev.open and cur.open <= prev.close:
            ctx.price_action_pattern = "BULLISH_ENGULFING"
        elif cur.close < cur.open and prev.close > prev.open and cur.close <= prev.open and cur.open >= prev.close:
            ctx.price_action_pattern = "BEARISH_ENGULFING"
        else:
            ctx.price_action_pattern = "NONE"

    # VOLATILITY_COMPRESSION_EXPANSION
    short_atr = _avg_true_range(bars, VOLATILITY_SHORT_WINDOW)
    long_atr = _avg_true_range(bars, VOLATILITY_LONG_WINDOW)
    if short_atr is not None and long_atr is not None and long_atr > 0:
        ratio = short_atr / long_atr
        ctx.atr_ratio = ratio
        ctx.volatility_state = ("COMPRESSION" if ratio < VOLATILITY_COMPRESSION_RATIO else
                                 "EXPANSION" if ratio > VOLATILITY_EXPANSION_RATIO else "NORMAL")

    # VOLUME / RELATIVE_VOLUME
    ctx.latest_volume = latest.volume
    if latest.volume is not None and len(bars) > RELATIVE_VOLUME_LOOKBACK:
        prior_volumes = [b.volume for b in bars[-RELATIVE_VOLUME_LOOKBACK - 1:-1] if b.volume is not None]
        if prior_volumes:
            avg_vol = sum(prior_volumes) / len(prior_volumes)
            if avg_vol > 0:
                rvol = latest.volume / avg_vol
                ctx.relative_volume_raw = rvol
                ctx.relative_volume_state = ("HIGH_RVOL" if rvol > RELATIVE_VOLUME_HIGH else
                                              "LOW_RVOL" if rvol < RELATIVE_VOLUME_LOW else "NORMAL_RVOL")

    # VWAP (disclosed daily-bar approximation - see ontology.py)
    if len(bars) >= VWAP_PROXY_LOOKBACK:
        window_bars = [b for b in bars[-VWAP_PROXY_LOOKBACK:] if b.volume]
        total_vol = sum(b.volume for b in window_bars)
        if total_vol > 0:
            vwap_proxy = sum(((b.high + b.low + b.close) / 3) * b.volume for b in window_bars) / total_vol
            ctx.vwap_proxy_20d = vwap_proxy
            if vwap_proxy > 0:
                dist = (latest.close - vwap_proxy) / vwap_proxy
                ctx.vwap_state = ("ABOVE_VWAP_PROXY" if dist > VWAP_PROXY_THRESHOLD else
                                   "BELOW_VWAP_PROXY" if dist < -VWAP_PROXY_THRESHOLD else "AT_VWAP_PROXY")

    # MOVING_AVERAGE_STRUCTURE
    mas = [_sma(closes, p) for p in MA_STACK_PERIODS]
    if all(m is not None for m in mas):
        ma20, ma50, ma200 = mas
        if ma20 > ma50 > ma200:
            ctx.ma_stack = "BULLISH_STACK"
        elif ma20 < ma50 < ma200:
            ctx.ma_stack = "BEARISH_STACK"
        else:
            ctx.ma_stack = "MIXED_STACK"

    # MARKET_STRUCTURE (two consecutive halves of a fixed window - see module docstring)
    if len(bars) >= 2 * MARKET_STRUCTURE_HALF_WINDOW:
        window = bars[-2 * MARKET_STRUCTURE_HALF_WINDOW:]
        first_half, second_half = window[:MARKET_STRUCTURE_HALF_WINDOW], window[MARKET_STRUCTURE_HALF_WINDOW:]
        first_high, first_low = max(b.high for b in first_half), min(b.low for b in first_half)
        second_high, second_low = max(b.high for b in second_half), min(b.low for b in second_half)
        if second_high > first_high and second_low > first_low:
            ctx.market_structure = "HIGHER_HIGHS_HIGHER_LOWS"
        elif second_high < first_high and second_low < first_low:
            ctx.market_structure = "LOWER_HIGHS_LOWER_LOWS"
        else:
            ctx.market_structure = "MIXED"

    # GAPS
    if len(bars) >= 2:
        prev_close = bars[-2].close
        if prev_close > 0:
            gap = (latest.open - prev_close) / prev_close
            ctx.gap_pct = gap
            ctx.gap_state = ("GAP_UP" if gap > GAP_THRESHOLD else
                              "GAP_DOWN" if gap < -GAP_THRESHOLD else "NO_GAP")

    # RELATIVE_STRENGTH (vs. benchmark)
    if len(closes) > RELATIVE_STRENGTH_LOOKBACK:
        entity_return = (closes[-1] - closes[-RELATIVE_STRENGTH_LOOKBACK - 1]) / closes[-RELATIVE_STRENGTH_LOOKBACK - 1]
        bench_bars = ohlcv.bars(benchmark_ticker, as_of, lookback_days=MAX_LOOKBACK_CALENDAR_DAYS)
        bench_closes = [b.close for b in bench_bars]
        if len(bench_closes) > RELATIVE_STRENGTH_LOOKBACK:
            bench_return = ((bench_closes[-1] - bench_closes[-RELATIVE_STRENGTH_LOOKBACK - 1]) /
                             bench_closes[-RELATIVE_STRENGTH_LOOKBACK - 1])
            rs = entity_return - bench_return
            ctx.relative_strength_20d = rs
            ctx.relative_strength_state = ("OUTPERFORMING" if rs > RELATIVE_STRENGTH_THRESHOLD else
                                            "UNDERPERFORMING" if rs < -RELATIVE_STRENGTH_THRESHOLD else "IN_LINE")

    # CLOSE_LOCATION_VALUE (stage 7 item 7) - averaged over a window since a single bar's value is
    # too noisy to use as a stable state on its own (see ontology.py's CONCEPT_REGISTRY entry).
    if len(bars) >= CLOSE_LOCATION_VALUE_LOOKBACK:
        window_bars = bars[-CLOSE_LOCATION_VALUE_LOOKBACK:]
        clv_values = [((b.close - b.low) - (b.high - b.close)) / (b.high - b.low)
                      for b in window_bars if b.high > b.low]
        if clv_values:
            avg_clv = sum(clv_values) / len(clv_values)
            ctx.clv_raw = avg_clv
            ctx.clv_state = ("UPPER_RANGE" if avg_clv > CLV_UPPER_THRESHOLD else
                              "LOWER_RANGE" if avg_clv < CLV_LOWER_THRESHOLD else "MID_RANGE")

    # LIQUIDITY_REGIME (stage 7 item 7) - self-relative dollar-volume level, NEVER a cross-security
    # comparison (see ontology.py's CONCEPT_REGISTRY entry and module docstring).
    if len(bars) >= LIQUIDITY_LONG_WINDOW:
        long_window_bars = bars[-LIQUIDITY_LONG_WINDOW:]
        long_dollar_volumes = [b.close * b.volume for b in long_window_bars if b.volume is not None]
        short_dollar_volumes = [b.close * b.volume for b in long_window_bars[-LIQUIDITY_SHORT_WINDOW:]
                                 if b.volume is not None]
        if long_dollar_volumes and short_dollar_volumes:
            long_avg = sum(long_dollar_volumes) / len(long_dollar_volumes)
            short_avg = sum(short_dollar_volumes) / len(short_dollar_volumes)
            if long_avg > 0:
                ratio = short_avg / long_avg
                ctx.liquidity_dollar_volume_ratio = ratio
                ctx.liquidity_regime = ("HIGH_LIQUIDITY" if ratio > LIQUIDITY_HIGH_RATIO else
                                         "LOW_LIQUIDITY" if ratio < LIQUIDITY_LOW_RATIO else "NORMAL_LIQUIDITY")

    # MULTI_TIMEFRAME_CONFIRMATION (daily trend vs. a weekly-resampled trend from the same series)
    weekly_closes = closes[::-5][::-1]  # every 5th trading day, oldest-first - a real-bar weekly proxy
    weekly_trend = _trend_from_closes(weekly_closes) if len(weekly_closes) >= TREND_MA_PERIOD + TREND_SLOPE_LOOKBACK else "UNKNOWN"
    if ctx.trend_direction in ("UP", "DOWN") and weekly_trend in ("UP", "DOWN"):
        ctx.mtf_confirmation = "CONFIRMED" if ctx.trend_direction == weekly_trend else "CONFLICTED"
    else:
        ctx.mtf_confirmation = "UNKNOWN"

    return ctx


# The DISCRETE, matchable technical-concept fields - never the raw diagnostic floats (those aren't
# used for conditioning/matching). Canonical, shared source for both experiment/context.py (which
# merges these into the live context dict) and learn/hypothesis.py (which proposes conditions over
# them) - defined once here rather than duplicated, so the two can never silently drift apart.
TECHNICAL_STATE_FIELD_NAMES: list[str] = [
    "trend_direction", "momentum_state", "mean_reversion_state", "breakout_state", "pullback_state",
    "price_action_pattern", "support_resistance_state", "volatility_state", "relative_volume_state",
    "vwap_state", "ma_stack", "market_structure", "gap_state", "relative_strength_state", "mtf_confirmation",
    "clv_state", "ma_slope_state", "liquidity_regime",  # stage 7 item 7
]

# Maps each technical field name back to the canonical concept it operationalizes - used for
# methodology/concept provenance (learn/hypothesis.py records which TradingConcept(s) a proposed
# hypothesis touches, and looks up which methodologies independently mapped to that same concept).
DIMENSION_TO_CONCEPT: dict[str, TradingConcept] = {
    "trend_direction": TradingConcept.TREND,
    "momentum_state": TradingConcept.MOMENTUM,
    "mean_reversion_state": TradingConcept.MEAN_REVERSION,
    "breakout_state": TradingConcept.BREAKOUT,
    "pullback_state": TradingConcept.PULLBACK,
    "price_action_pattern": TradingConcept.PRICE_ACTION,
    "support_resistance_state": TradingConcept.SUPPORT_RESISTANCE,
    "volatility_state": TradingConcept.VOLATILITY_COMPRESSION_EXPANSION,
    "relative_volume_state": TradingConcept.RELATIVE_VOLUME,
    "vwap_state": TradingConcept.VWAP,
    "ma_stack": TradingConcept.MOVING_AVERAGE_STRUCTURE,
    "market_structure": TradingConcept.MARKET_STRUCTURE,
    "gap_state": TradingConcept.GAPS,
    "relative_strength_state": TradingConcept.RELATIVE_STRENGTH,
    "mtf_confirmation": TradingConcept.MULTI_TIMEFRAME_CONFIRMATION,
    "clv_state": TradingConcept.CLOSE_LOCATION_VALUE,
    "ma_slope_state": TradingConcept.TREND,  # a more granular measurement of TREND, not a new concept
    "liquidity_regime": TradingConcept.LIQUIDITY_REGIME,
}

# Each technical field's "uninteresting"/default value(s) - e.g. a FLAT trend or a NORMAL volatility
# state is not a distinguishing condition. Canonical, shared source (stage 7) for learn/hypothesis.py's
# per-value conditioning dimensions AND learn/hierarchical_research.py's Level 1 family-screening
# query ("any value NOT in this set") - defined once so the two can never silently disagree about
# what counts as "nothing interesting happening" for a given field.
TECHNICAL_DEFAULT_VALUES: dict[str, tuple[str, ...]] = {
    "trend_direction": ("UNKNOWN", "FLAT"),
    "momentum_state": ("UNKNOWN", "FLAT"),
    "breakout_state": ("UNKNOWN", "NONE"),
    "mean_reversion_state": ("UNKNOWN", "NORMAL"),
    "pullback_state": ("UNKNOWN", "NONE"),
    "volatility_state": ("UNKNOWN", "NORMAL"),
    "relative_volume_state": ("UNKNOWN", "NORMAL_RVOL"),
    "relative_strength_state": ("UNKNOWN", "IN_LINE"),
    "ma_stack": ("UNKNOWN", "MIXED_STACK"),
    "market_structure": ("UNKNOWN", "MIXED"),
    "gap_state": ("UNKNOWN", "NO_GAP"),
    "vwap_state": ("UNKNOWN", "AT_VWAP_PROXY"),
    "support_resistance_state": ("UNKNOWN", "MID_RANGE"),
    "price_action_pattern": ("UNKNOWN", "NONE"),
    "mtf_confirmation": ("UNKNOWN",),
    "clv_state": ("UNKNOWN", "MID_RANGE"),
    "ma_slope_state": ("UNKNOWN", "FLAT"),
    "liquidity_regime": ("UNKNOWN", "NORMAL_LIQUIDITY"),
}
