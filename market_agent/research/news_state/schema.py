"""The News State Engine's data model - article -> EVENT -> COMPANY STATE,
kept as three deliberately distinct objects (never conflated - see this
package's module docstring in event_vector.py for why).

THIS IS A STRUCTURED ECONOMIC REPRESENTATION, NOT A SENTIMENT SCORE. Every
IMPLICATION_AXIS below is a SEPARATE, independently-populated, signed
quantity - "layoffs due to declining demand" can and should show
demand<0, risk>0, profitability>=0 simultaneously. Collapsing these into
one number is exactly the failure mode this module exists to avoid.

EVERY NUMBER HAS A DISCLOSED PROVENANCE: `EventVector`/`CompanyNewsState`
never silently mix "confidently observed" with "guessed" - see
epistemic.py for the axis that encodes this, and llm_schema.py for the
explicit, versioned boundary between the deterministic rule-based
extractor (what actually runs in this environment) and an LLM-backed one
(designed, not claimed as active unless a real client is configured).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The ten economic-implication axes this design settled on - see the validation report for why this
# set and not a shorter/longer one. Each is a SIGNED scalar: positive = improving/favorable for the
# company, negative = deteriorating/unfavorable, None = this event/state says nothing about this axis
# (never coerced to 0.0 - zero would mean "confidently neutral", which is a different claim).
IMPLICATION_AXES: tuple[str, ...] = (
    "growth", "profitability", "cash_flow", "balance_sheet", "demand", "supply_chain",
    "competitive_position", "regulatory", "guidance", "risk",
)
# risk's sign convention is the one deliberate exception: positive = risk INCREASING (elevated),
# negative = risk DECREASING (abating) - "higher number = more of the named thing" for every axis,
# and for risk, more of the named thing means more risk, not more favorability. Documented once here
# rather than re-derived at every call site.

EPISTEMIC_STATUSES: tuple[str, ...] = (
    "OBSERVED_FACT", "MANAGEMENT_CLAIM", "THIRD_PARTY_REPORTING", "ANALYST_INTERPRETATION", "SPECULATION",
)
# Ordered strongest -> weakest evidentiary weight - see epistemic.py's CERTAINTY_WEIGHT for the
# numeric mapping this order implies.

TIME_HORIZONS: tuple[str, ...] = ("SHORT_TERM", "MEDIUM_TERM", "LONG_TERM", "UNSPECIFIED")


@dataclass
class EventVector:
    """Layer B - "what economically relevant event happened" - built from
    ONE cluster of deduplicated TimelineEvents describing the SAME
    underlying occurrence (see event_vector.py). NOT one row per article -
    the whole point of clustering first is that 30 syndicated copies of
    the same story produce ONE EventVector, not 30."""
    event_vector_id: str
    entity: str
    as_of: str                              # ISO - the LATEST constituent clause's date
    description: str
    implications: dict[str, float | None]     # IMPLICATION_AXES -> signed value in [-1,1] or None
    text_sentiment: float | None                 # separate, non-authoritative - lexicon polarity, [-1,1]
    materiality: float                             # [0,1], derived from the existing HIGH/MEDIUM/LOW tag
    certainty: float                                 # [0,1] - see epistemic.py; a DISCLOSED heuristic,
    #                                                    never a fitted/calibrated probability (see llm_schema.py)
    epistemic_status: str                              # dominant status among constituent clauses
    implication_basis: dict[str, str | None] = field(default_factory=dict)  # IMPLICATION_AXES ->
    #    "STATED" (the clause directly named this axis - a DIRECT IMPLICATION_RULES hit), "INFERRED"
    #    (this axis was DERIVED from a clause about something else - an INVERSE rule hit, or the fixed
    #    cost-reduction->profitability rule; e.g. "risk" is ALWAYS inferred, never directly stated by a
    #    news clause), "MIXED" (both kinds contributed), or None (axis not populated at all). See
    #    event_vector.py's _implications_from_events docstring - this makes explicit a fact/inference
    #    boundary that the DIRECT/INVERSE relationship type always encoded internally but never surfaced.
    dispersion: dict[str, float | None] = field(default_factory=dict)  # WITHIN-event variance per axis -
    #     when this EventVector's own constituent clauses disagree on an axis (found necessary: two
    #     opposing clauses clustered into one narrative were silently averaged to 0.0 with the
    #     disagreement discarded) - see aggregation.py's law-of-total-variance combination with
    #     BETWEEN-event dispersion at the company-state level.
    epistemic_breakdown: dict[str, int] = field(default_factory=dict)   # status -> clause count
    time_horizon: str = "UNSPECIFIED"
    independent_source_count: int = 0
    source_quality: float = 0.0                          # [0,1], weighted-average reliability tier
    confirmation_strength: float = 0.0                      # [0,1], SATURATING function of independent
    #                                                          confirmation - see aggregation.py; this is what
    #                                                          keeps confidence (not magnitude) rising with N sources
    novelty: float | None = None                              # None until a historical baseline exists
    #                                                             for this (entity, implication-shape) - see
    #                                                             aggregation.py's novelty scaffold
    magnitude_confidence: float = 0.0                            # [0,1] - what fraction of this event's
    #                                                                own implication contributions came from a
    #                                                                genuinely extracted, anchored magnitude
    #                                                                (magnitude.py) rather than a direction-only
    #                                                                fallback - 0.0 if it never fired any rule at all
    constituent_event_ids: list[str] = field(default_factory=list)
    constituent_source_ids: list[str] = field(default_factory=list)
    extraction_method: str = "RULE_BASED_V1"                    # versioned - see llm_schema.py

    def to_dict(self) -> dict:
        return {
            "event_vector_id": self.event_vector_id, "entity": self.entity, "as_of": self.as_of,
            "description": self.description, "implications": self.implications,
            "text_sentiment": self.text_sentiment, "materiality": self.materiality,
            "certainty": self.certainty, "epistemic_status": self.epistemic_status,
            "implication_basis": self.implication_basis,
            "dispersion": self.dispersion,
            "epistemic_breakdown": self.epistemic_breakdown, "time_horizon": self.time_horizon,
            "independent_source_count": self.independent_source_count, "source_quality": self.source_quality,
            "confirmation_strength": self.confirmation_strength, "novelty": self.novelty,
            "magnitude_confidence": self.magnitude_confidence,
            "constituent_event_ids": self.constituent_event_ids,
            "constituent_source_ids": self.constituent_source_ids, "extraction_method": self.extraction_method,
        }


@dataclass
class CompanyNewsState:
    """Layer C - N(t): the decay/confirmation-weighted aggregation of many
    EventVectors into one per-company, per-timestamp state. See
    aggregation.py for exactly how each field below is computed."""
    entity: str
    as_of: str
    dimensions: dict[str, float | None]           # IMPLICATION_AXES -> weighted mean, None if no event fed it
    dispersion: dict[str, float | None]              # IMPLICATION_AXES -> weighted variance among contributing
    #                                                    events - a CONTRADICTION/uncertainty signal, not noise to discard
    text_sentiment: float | None = None
    confidence: float = 0.0                              # [0,1] heuristic overall confidence - see aggregation.py
    news_volume: int = 0                                    # RAW document count (pre-dedup) - disclosed as such,
    #                                                          never used as an aggregation WEIGHT (see module docstring)
    independent_event_count: int = 0                          # number of distinct EventVectors contributing
    dominant_event_ids: list[str] = field(default_factory=list)   # top-materiality contributing events
    contradiction_axes: list[str] = field(default_factory=list)     # axes whose dispersion exceeds threshold
    source_quality: float = 0.0
    state_change: dict[str, float] | None = None                       # ΔN per axis vs. the prior persisted state
    state_velocity: float | None = None                                   # ||ΔN|| / Δt (only if a prior state exists)
    state_direction: dict[str, float] | None = None                         # unit vector of ΔN
    half_life_days: float = 7.0                                                # decay parameter actually used - see
    #                                                                             aggregation.py's justification
    excluded_by_role: dict[str, int] = field(default_factory=dict)                 # entity_resolution.py -
    #   count of clauses SEEN but excluded from every field above because they were attributed to a
    #   different real-world party (COMPETITOR/INDUSTRY - never COUNTERPARTY/SUBSIDIARY, which stay
    #   attributable), keyed by role. Visible so exclusion is disclosed, not a silent drop - see
    #   entity_resolution.py's module docstring.

    def to_dict(self) -> dict:
        return {
            "entity": self.entity, "as_of": self.as_of, "dimensions": self.dimensions,
            "dispersion": self.dispersion, "text_sentiment": self.text_sentiment, "confidence": self.confidence,
            "news_volume": self.news_volume, "independent_event_count": self.independent_event_count,
            "dominant_event_ids": self.dominant_event_ids, "contradiction_axes": self.contradiction_axes,
            "source_quality": self.source_quality, "state_change": self.state_change,
            "state_velocity": self.state_velocity, "state_direction": self.state_direction,
            "half_life_days": self.half_life_days, "excluded_by_role": self.excluded_by_role,
        }
