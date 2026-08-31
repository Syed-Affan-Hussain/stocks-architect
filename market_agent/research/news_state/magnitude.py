"""Magnitude extraction and anchored scoring - replaces the flat +-1.0
every prior implication used, regardless of whether the underlying text
said "grew 2%" or "grew 40%".

REGEX-BASED, NOT NLP: this finds numbers with a recognizable financial
unit (%, bps, $) near the clause under consideration. It does NOT do
dependency parsing to work out which of several numbers in a multi-metric
sentence belongs to which claim - see extract_primary_magnitude's own
docstring for the disclosed simplification this relies on instead (the
existing clause splitter already isolates most multi-metric sentences on
its own, since it already splits on contrast conjunctions - "revenue grew
40%, but margins fell 200bps" is already two clauses by the time this
module ever sees it).

ANCHORS ARE FIXED AND DISCLOSED, NOT FITTED: there is no labeled dataset
in this project mapping "a human financial analyst would call this move
small/moderate/large" to a specific percentage - the anchor points below
are a stated, reasoned convention (0% = no effect, 5% = a real but modest
move, 15% = a clearly material move, 30%+ = an exceptional move beyond
which finer distinctions aren't reliably meaningful from text alone), not
a calibrated fit. Changing them is a disclosed, versioned decision.

USD MAGNITUDES ARE EXTRACTED BUT NOT SCORED: "what counts as a large
dollar figure" depends entirely on the company's own scale (a $500M
guidance raise means something different for a $10B company than for a
$500B one), and this module has no company-size context to normalize
against. Rather than inventing an arbitrary universal dollar anchor, a
detected dollar figure is reported (for audit/display) and the event
falls back to DIRECTION_ONLY scoring - a disclosed limitation, not a
silent gap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)", re.IGNORECASE)
BPS_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:bps|basis points?)", re.IGNORECASE)
DOLLAR_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*(billion|million|bn|mn|b|m)\b", re.IGNORECASE)
DOLLAR_UNIT_MULTIPLIER = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6}

# Fixed, disclosed anchor points - (magnitude, score) pairs, piecewise-linear interpolated between
# them and clamped beyond the last one. See module docstring for the reasoning behind each anchor.
PERCENT_ANCHORS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (5.0, 0.3), (15.0, 0.6), (30.0, 1.0))
BPS_ANCHORS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (50.0, 0.3), (150.0, 0.6), (300.0, 1.0))
DIRECTION_ONLY_SCORE = 0.5  # the deliberate MIDPOINT between "small" (0.3) and "moderate" (0.6) on
#   the percent anchor scale - "real but unmeasured", never biased toward looking small OR large just
#   because no specific number was stated in the text.

MAGNITUDE_CONFIDENCE = {"PERCENT": 0.85, "BPS": 0.85, "USD": 0.5, "DIRECTION_ONLY": 0.4}
#   USD sits at 0.5 (not 0.85) because it is extracted but NOT scored - a real number was found, but
#   it doesn't yet inform the magnitude, so it shouldn't claim the same confidence as a scored one.


@dataclass(frozen=True)
class MagnitudeFact:
    raw_text: str
    value: float          # ALWAYS as extracted from the text, sign as written (may be unsigned)
    unit: str              # "PERCENT" | "BPS" | "USD"

    def to_dict(self) -> dict:
        return {"raw_text": self.raw_text, "value": self.value, "unit": self.unit}


def extract_magnitudes(text: str) -> list[MagnitudeFact]:
    """Every recognizable numeric magnitude in `text`, in the order
    found. A clause can contain more than one (rare, but not
    prevented) - see extract_primary_magnitude for how one is chosen
    when the caller needs a single representative value."""
    facts: list[MagnitudeFact] = []
    for m in PERCENT_RE.finditer(text):
        facts.append(MagnitudeFact(raw_text=m.group(0), value=float(m.group(1)), unit="PERCENT"))
    for m in BPS_RE.finditer(text):
        facts.append(MagnitudeFact(raw_text=m.group(0), value=float(m.group(1)), unit="BPS"))
    for m in DOLLAR_RE.finditer(text):
        multiplier = DOLLAR_UNIT_MULTIPLIER.get(m.group(2).lower(), 1.0)
        facts.append(MagnitudeFact(raw_text=m.group(0), value=float(m.group(1)) * multiplier, unit="USD"))
    return facts


def extract_primary_magnitude(text: str) -> MagnitudeFact | None:
    """The single most usable magnitude in a clause - PERCENT preferred
    over BPS over USD (percent is the most directly comparable/scoreable
    unit; USD is extracted but never scored - see module docstring). Does
    NOT attempt to disambiguate which of several same-unit numbers in one
    clause belongs to which claim; the FIRST one found (reading order) is
    used, which is a real, disclosed simplification - clauses with
    multiple same-unit numbers are rare after the existing contrast-
    conjunction clause split (a sentence naming two percentages for two
    different claims usually contains a "but"/"while" that already split
    it into separate clauses before this function ever runs)."""
    facts = extract_magnitudes(text)
    if not facts:
        return None
    priority = {"PERCENT": 0, "BPS": 1, "USD": 2}
    facts_sorted = sorted(facts, key=lambda f: priority.get(f.unit, 9))
    best_unit = facts_sorted[0].unit
    return next(f for f in facts if f.unit == best_unit)


def _interpolate(anchors: tuple[tuple[float, float], ...], x: float) -> float:
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return anchors[-1][1]  # clamp beyond the last anchor


def magnitude_to_score(fact: MagnitudeFact) -> float | None:
    """|fact.value| mapped through the fixed anchor table for its unit,
    in [0, 1] - the CALLER applies sign (see event_vector.py: the
    lexicon-derived direction, not this function, decides +/-, since a
    plain "5%" extracted from text carries no direction of its own unless
    the number itself was written with an explicit minus sign). Returns
    None for USD (not scored - see module docstring) or an unrecognized
    unit."""
    if fact.unit == "PERCENT":
        return round(_interpolate(PERCENT_ANCHORS, abs(fact.value)), 2)
    if fact.unit == "BPS":
        return round(_interpolate(BPS_ANCHORS, abs(fact.value)), 2)
    return None  # USD: extracted, not scored


def explicit_sign(fact: MagnitudeFact) -> int | None:
    """+1/-1 if the extracted text carried an explicit sign (e.g. "-5%"
    or "5% decline" would NOT count - only a literal minus sign attached
    to the number does); None if the number itself is unsigned, meaning
    the caller must supply direction from elsewhere (the lexicon)."""
    if fact.raw_text.strip().startswith("-"):
        return -1
    return None
