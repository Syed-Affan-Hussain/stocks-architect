"""A small, hand-curated, PROOF-OF-CONCEPT seed corpus of publicly
documented, well-established trading methodologies - NOT the "broad
corpus... including lesser-known traders and related practitioners" the
methodology-ingestion layer was asked to cover. That gap is disclosed
here explicitly, not glossed over:

  A genuinely broad-corpus ingestion, especially one reaching lesser-known
  practitioners, needs one of two things this environment doesn't safely
  have: (1) an LLM capable of reading arbitrary long-form text (books,
  interviews, blog posts) at scale and mapping it onto the ontology with
  real generality - see extractor.py's LLMMethodologyExtractor, left
  unimplemented for the same reason every other LLM slot in this project
  is (no SDK/API key configured, verified directly); or (2) extensive
  manual research and verification per practitioner, which risks
  misattributing specific claims to a real person if done carelessly at
  the volume "broad corpus" implies. Neither is something this session can
  responsibly do at scale.

  What IS in scope, and is what this module provides: a small set of
  WELL-ESTABLISHED, EXTENSIVELY PUBLICLY DOCUMENTED methodologies (each
  with its own widely available published book, research paper, or
  long-running public track record of description by the practitioner
  themselves) - never obscure or unverifiable, and every description below
  is a SHORT PARAPHRASE IN THIS PROJECT'S OWN WORDS, never a verbatim
  reproduction of the source material. This exists to prove the ingestion
  pipeline (extractor -> store -> hypothesis generator -> governed testing)
  actually works end to end against real, attributable methodology
  content - not to claim comprehensive corpus coverage.

NO PROFITABILITY CLAIM IS EVER CAPTURED FROM ANY OF THESE - see
methodology/schema.py's module docstring: only a concept mapping is
recorded, never a stated win rate or return.
"""
from __future__ import annotations

from market_agent.methodology.schema import RawMethodologySource

SEED_CORPUS: list[RawMethodologySource] = [
    RawMethodologySource(
        name="Darvas Box", practitioner="Nicolas Darvas", source_type="book",
        raw_text="A price-consolidation 'box' framework: a stock is watched while it trades in a tight "
                 "range, then a breakout above the box high on a volume surge is treated as a trend "
                 "continuation signal, with the box's own high acting as a support/resistance reference "
                 "for managing the position afterward."),
    RawMethodologySource(
        name="Turtle Trading Rules", practitioner="Richard Dennis / Curtis Faith", source_type="book",
        raw_text="A trend-following breakout system: entries trigger on a breakout beyond a recent price "
                 "channel high or low, position size is set from a volatility contraction/expansion "
                 "measure of the instrument's own recent range, and stop placement follows an explicit "
                 "risk-per-trade / position sizing rule."),
    RawMethodologySource(
        name="CANSLIM", practitioner="William O'Neil", source_type="book",
        raw_text="A growth-stock screening and timing approach emphasizing relative strength leadership "
                 "versus the broader market, confirmation via a volume surge on the breakout day, and "
                 "sector leadership - looking for stocks whose industry group and market structure both "
                 "show a confirmed uptrend."),
    RawMethodologySource(
        name="SEPA / Trend Template", practitioner="Mark Minervini", source_type="book",
        raw_text="A trend-template checklist built on moving average stack alignment (shorter-period "
                 "averages above longer-period averages), relative strength versus the market, and entries "
                 "on a shallow pullback or a volume-confirmed breakout once the trend template criteria "
                 "are met."),
    RawMethodologySource(
        name="Bollinger Bands Mean Reversion", practitioner="John Bollinger", source_type="book",
        raw_text="A volatility-band framework where price reaching the outer bands after a period of "
                 "volatility contraction (a 'squeeze') is treated as a candidate mean reversion or "
                 "volatility expansion setup, depending on how price behaves at the band."),
    RawMethodologySource(
        name="VWAP Institutional Reversion", practitioner="widely documented intraday trading practice",
        source_type="publicly_documented_system",
        raw_text="An intraday approach used by institutional execution desks and short-term traders: price "
                 "trading meaningfully away from the volume-weighted average price (VWAP) on relative "
                 "volume that is elevated versus its own average is treated as a candidate reversion-toward-"
                 "VWAP setup."),
    RawMethodologySource(
        name="Opening Range Breakout (ORB)", practitioner="Toby Crabel", source_type="published_research",
        raw_text="A short-term breakout approach: a volatility contraction in the days before the session, "
                 "combined with a gap or a breakout beyond the opening range established in the first "
                 "minutes of trading, is treated as a signal for continuation in the breakout direction."),
    RawMethodologySource(
        name="Classic Price-Action / Support-Resistance Trading", practitioner="general technical-analysis "
        "tradition, not attributed to one individual", source_type="publicly_documented_system",
        raw_text="A discretionary approach built on candlestick price action - engulfing bars, inside bars, "
                 "and outside bars - read at prior support and resistance levels, with a setup considered "
                 "stronger when the same signal is confirmed across multiple timeframes (e.g. daily and "
                 "weekly)."),
    RawMethodologySource(
        name="Catalyst / Event-Driven Trading", practitioner="general event-driven trading practice, not "
        "attributed to one individual", source_type="publicly_documented_system",
        raw_text="A short-term approach centered on a specific dated catalyst - an earnings reaction or "
                 "other news-driven event - where a resulting price gap and a volume surge above the "
                 "instrument's average are treated as a tradable event reaction rather than a technical "
                 "pattern."),
]
