"""The canonical Trading Concept ontology - stage 6's shared vocabulary
that both the technical-context provider (technical_context.py) and the
methodology-ingestion layer (market_agent/methodology/) map onto. Every
methodology this system ever ingests, no matter which trader or source it
comes from, gets expressed as ONE OR MORE of these fixed concepts - so
two traders independently describing "buy the first pullback to the rising
20-day average" reinforce the SAME concept (TREND + MEAN_REVERSION,
concretely), not two duplicate strategies. See methodology/schema.py for
how provenance (which trader/methodology contributed a mapping) is kept
separate from the concept itself.

FIXED, NOT EXTENSIBLE PER-HYPOTHESIS: this is a closed enum, matching the
blueprint's own "candidate explanatory variables, not asserted signal"
discipline (see events/schema.py's ContextSnapshot docstring for the
precedent) - a new concept is added here as a disclosed, versioned
decision, never invented ad hoc by a hypothesis generator. STAGE 7 ITEM 7
added CLOSE_LOCATION_VALUE and LIQUIDITY_REGIME (20 concepts became 22) as
exactly such a disclosed, versioned decision - see technical_context.py's
module docstring for what each is built from. Moving-average slope, the
third stage-7-item-7 addition, is exposed as a NEW field
(`ma_slope_state`) but deliberately maps back onto the EXISTING TREND
concept rather than spawning its own - it is a more granular measurement
of the same underlying phenomenon TREND already names (steepness, not a
different market characteristic), not a new one.

WHICH CONCEPTS THIS SYSTEM CAN ACTUALLY COMPUTE, DISCLOSED HONESTLY: 19 of
22 are computable from data this system actually has (daily OHLCV + a
benchmark series + the existing event-interpretation pipeline) - marked
`computable=True` below, each with a one-line note on what it's actually
built from and any approximation involved. THREE ARE NOT, and are marked
`computable=False` rather than faked:

  OPENING_RANGE - genuinely requires intraday/minute-resolution bars to
    define "the first N minutes of the session". This system only has
    daily OHLCV (see outcomes/ohlcv.py) - there is no sub-day resolution
    to compute an opening range FROM. Not approximated; left honestly
    absent from TechnicalMarketContext.
  SECTOR_CONTEXT - genuinely requires a sector classification/sector-index
    data source this system has never had wired in (events/schema.py's
    ContextSnapshot docstring already discloses this same gap for
    `sector_momentum` - stage 6 does not close it, just names it formally
    here so the ontology is honest about its own limits).
  RISK_MANAGEMENT - not a market-STATE signal the way the other concepts
    are. It describes how a methodology sizes positions and places stops,
    which isn't observable from price/volume data at all - it's tracked at
    the METHODOLOGY-PROVENANCE level (a methodology can BE about risk
    management) but is never a conditioning variable in the hypothesis
    generator, because testing "does this stop-loss rule improve abnormal
    returns" would require simulating actual trade entries/exits/sizing,
    a fundamentally different (and much larger) kind of backtest than this
    system's event-study design does.

TWO FURTHER GAPS, NAMED HERE RATHER THAN INSIDE THE ENUM: stage 7 item 7
explicitly asked that unavailable variables be marked, not manufactured.
Market breadth (the proportion of stocks/sectors participating in a move -
e.g. advance/decline data) and cross-security correlation (how one
entity's price co-moves with others beyond its single benchmark) are
NEITHER represented as a TradingConcept NOR as a TechnicalMarketContext
field, because this system has never had a market-breadth data source or a
cross-security correlation matrix wired in - only per-entity daily OHLCV
plus one benchmark series (outcomes/ohlcv.py, experiment/context.py). They
are not folded into RELATIVE_STRENGTH (which is a real, computed,
single-benchmark comparison) or SECTOR_CONTEXT (which is already the
disclosed sector-level gap above) because both would misrepresent a
market-wide or cross-security signal as something this system actually
measures. Intraday-only signals (true VWAP, a real opening range, and any
candle pattern finer than daily resolution) are the SAME gap already
disclosed under VWAP/OPENING_RANGE/PRICE_ACTION below - restated here only
because item 7 asked for an explicit list, not because it is new.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TradingConcept(str, Enum):
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    PRICE_ACTION = "PRICE_ACTION"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    VOLATILITY_COMPRESSION_EXPANSION = "VOLATILITY_COMPRESSION_EXPANSION"
    VOLUME = "VOLUME"
    RELATIVE_VOLUME = "RELATIVE_VOLUME"
    VWAP = "VWAP"
    MOVING_AVERAGE_STRUCTURE = "MOVING_AVERAGE_STRUCTURE"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    GAPS = "GAPS"
    OPENING_RANGE = "OPENING_RANGE"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
    SECTOR_CONTEXT = "SECTOR_CONTEXT"
    MULTI_TIMEFRAME_CONFIRMATION = "MULTI_TIMEFRAME_CONFIRMATION"
    CATALYST_EVENT_REACTION = "CATALYST_EVENT_REACTION"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    CLOSE_LOCATION_VALUE = "CLOSE_LOCATION_VALUE"
    LIQUIDITY_REGIME = "LIQUIDITY_REGIME"


@dataclass(frozen=True)
class ConceptDefinition:
    concept: TradingConcept
    description: str
    computable: bool
    computation_note: str  # what it's built from, an approximation caveat, or why it's not computable


CONCEPT_REGISTRY: dict[TradingConcept, ConceptDefinition] = {
    TradingConcept.TREND: ConceptDefinition(
        TradingConcept.TREND, "Directional persistence in price over a lookback window.", True,
        "Slope/sign of a rolling simple-moving-average of daily closes (see technical_context.py)."),
    TradingConcept.MOMENTUM: ConceptDefinition(
        TradingConcept.MOMENTUM, "Rate of recent price change.", True,
        "N-day rate of change of daily closes."),
    TradingConcept.MEAN_REVERSION: ConceptDefinition(
        TradingConcept.MEAN_REVERSION, "Price displacement from a rolling average, as a candidate reversion signal.",
        True, "Z-score of the latest close against a rolling mean/stdev of daily closes."),
    TradingConcept.BREAKOUT: ConceptDefinition(
        TradingConcept.BREAKOUT, "Price clearing a recent range high/low.", True,
        "Latest close vs. the prior N-day rolling high/low (excluding the latest bar itself)."),
    TradingConcept.PULLBACK: ConceptDefinition(
        TradingConcept.PULLBACK, "A retracement against the prevailing trend.", True,
        "Retracement of the latest close from the rolling N-day high, conditioned on TREND being up."),
    TradingConcept.PRICE_ACTION: ConceptDefinition(
        TradingConcept.PRICE_ACTION, "Single/two-bar candle structure (inside bar, outside bar, engulfing).", True,
        "Rule-based comparison of the latest 1-2 daily bars' open/high/low/close - daily resolution only, "
        "no intraday candle shape."),
    TradingConcept.SUPPORT_RESISTANCE: ConceptDefinition(
        TradingConcept.SUPPORT_RESISTANCE, "Proximity to a recent swing high/low acting as a level.", True,
        "Distance of the latest close from the rolling N-day high/low, as a fraction of price."),
    TradingConcept.VOLATILITY_COMPRESSION_EXPANSION: ConceptDefinition(
        TradingConcept.VOLATILITY_COMPRESSION_EXPANSION, "Whether recent range/true-range is contracting or "
        "expanding relative to its own longer-run average.", True,
        "Ratio of a short-window average true range to a longer-window average true range, both from daily bars."),
    TradingConcept.VOLUME: ConceptDefinition(
        TradingConcept.VOLUME, "Raw traded volume level.", True, "Daily Volume, as reported by the data source."),
    TradingConcept.RELATIVE_VOLUME: ConceptDefinition(
        TradingConcept.RELATIVE_VOLUME, "Today's volume relative to its own recent average.", True,
        "Latest daily volume divided by a rolling N-day average daily volume."),
    TradingConcept.VWAP: ConceptDefinition(
        TradingConcept.VWAP, "Volume-weighted average price.", True,
        "DISCLOSED APPROXIMATION: an N-day volume-weighted average of daily typical price "
        "((high+low+close)/3), NOT true intraday VWAP - this system has no intraday/tick data. "
        "The gap between latest close and this proxy is what's exposed, never presented as real intraday VWAP."),
    TradingConcept.MOVING_AVERAGE_STRUCTURE: ConceptDefinition(
        TradingConcept.MOVING_AVERAGE_STRUCTURE, "Relative ordering ('stack') of multiple moving averages.", True,
        "Ordering of the 20/50/200-day simple moving averages (e.g. 20>50>200 = bullish stack)."),
    TradingConcept.MARKET_STRUCTURE: ConceptDefinition(
        TradingConcept.MARKET_STRUCTURE, "Higher-highs/higher-lows vs. lower-highs/lower-lows swing structure.",
        True, "Rule-based comparison of successive rolling-window swing highs/lows from daily bars - a "
        "real, disclosed simplification of full swing-pivot detection."),
    TradingConcept.GAPS: ConceptDefinition(
        TradingConcept.GAPS, "Discontinuity between one session's close and the next session's open.", True,
        "Latest daily open vs. the prior daily close."),
    TradingConcept.OPENING_RANGE: ConceptDefinition(
        TradingConcept.OPENING_RANGE, "Price behavior relative to the first N minutes of a session's range.",
        False, "NOT COMPUTABLE - requires intraday/minute-resolution bars this system does not have "
        "(see outcomes/ohlcv.py - only daily bars are available). Not approximated."),
    TradingConcept.RELATIVE_STRENGTH: ConceptDefinition(
        TradingConcept.RELATIVE_STRENGTH, "Entity return vs. a benchmark's return over the same window.", True,
        "Entity's N-day return minus the benchmark's N-day return, both from daily closes."),
    TradingConcept.SECTOR_CONTEXT: ConceptDefinition(
        TradingConcept.SECTOR_CONTEXT, "The entity's sector/industry-group's own trend and relative behavior.",
        False, "NOT COMPUTABLE - no sector classification or sector-index data source is wired into this "
        "system (same disclosed gap as events/schema.py ContextSnapshot's `sector_momentum` field)."),
    TradingConcept.MULTI_TIMEFRAME_CONFIRMATION: ConceptDefinition(
        TradingConcept.MULTI_TIMEFRAME_CONFIRMATION, "Agreement of a signal (e.g. trend) across more than one "
        "bar interval.", True, "Daily-bar trend direction compared against a weekly-resampled (5-trading-day) "
        "trend direction from the same underlying daily series."),
    TradingConcept.CATALYST_EVENT_REACTION: ConceptDefinition(
        TradingConcept.CATALYST_EVENT_REACTION, "Price reaction attributable to a specific dated event.", True,
        "Delegates to the EXISTING event-interpretation pipeline (events/schema.py's EventRecord/"
        "ContextSnapshot) rather than new price data - event_type, direction, and recency/crowding fields "
        "already captured there ARE this concept's operationalization."),
    TradingConcept.RISK_MANAGEMENT: ConceptDefinition(
        TradingConcept.RISK_MANAGEMENT, "Position sizing, stop placement, and reward:risk rules a methodology "
        "specifies.", False, "NOT a market-state signal - not observable from price/volume data, so never a "
        "TechnicalMarketContext field or a hypothesis-generator conditioning variable. Tracked only at the "
        "methodology-provenance level (methodology/schema.py) since methodologies legitimately describe it."),
    TradingConcept.CLOSE_LOCATION_VALUE: ConceptDefinition(
        TradingConcept.CLOSE_LOCATION_VALUE, "Where the close sits within each session's own high-low range - "
        "a proxy for buying/selling pressure within the bar, distinct from PRICE_ACTION's candle-shape rules.",
        True, "Per-bar ((close-low)-(high-close))/(high-low), averaged over a rolling N-day window (a single "
        "bar's value is too noisy to use as a stable state on its own)."),
    TradingConcept.LIQUIDITY_REGIME: ConceptDefinition(
        TradingConcept.LIQUIDITY_REGIME, "Whether an entity's OWN sustained trading liquidity is currently "
        "elevated or depressed relative to its own longer-run baseline - distinct from RELATIVE_VOLUME's "
        "single-day spike detection.", True, "Ratio of a short-window average daily dollar volume "
        "(close x volume) to a longer-window average - self-relative to the same entity's own history, "
        "NEVER a cross-security liquidity comparison (this system has no cross-security correlation/breadth "
        "data source - see this module's docstring)."),
}

COMPUTABLE_CONCEPTS: tuple[TradingConcept, ...] = tuple(
    c for c, d in CONCEPT_REGISTRY.items() if d.computable)
