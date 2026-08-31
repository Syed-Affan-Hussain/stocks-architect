"""Builds EventVector (schema.py) from a real event-instance cluster
(event_identity.py's cluster_events - NOT research/narratives.py's purely
topical grouping; see event_identity.py's module docstring for exactly
why event identity needed its own, stricter mechanism). Adds two things
clustering alone doesn't give: a genuinely MULTI-LABEL economic-
implication read of each constituent clause, and MAGNITUDE-AWARE scoring
in place of a flat +-1.0 (magnitude.py).

WHY RE-SCAN THE CLAUSES INSTEAD OF TRUSTING TimelineEvent.affected_area:
`affected_area` is single-label by construction (extraction.py picks ONE
area per clause). "Company announces layoffs due to declining demand" is
a real clause that touches BOTH "workforce" AND "demand" - a single-label
tag can only ever report one of them, silently dropping the other. This
module re-scans clause text for EVERY matching area, which is what lets
one clause populate risk (from the workforce trigger) and demand (from
the demand trigger) simultaneously - the exact "layoffs" example this
whole design exists to get right.

MAGNITUDE-AWARE SCORING (see magnitude.py for the anchor table and its
justification): for each clause, if a scoreable magnitude (a percent or
basis-point figure) can be extracted, the DIRECTION still comes from the
sentiment lexicon (or an explicit sign on the number itself), but the
SIZE of the contribution comes from the anchored magnitude table instead
of a flat 1.0. A clause with no extractable magnitude falls back to
DIRECTION_ONLY_SCORE (0.5) - real but unmeasured, never silently promoted
to look as certain as a quantified fact. `EventVector.magnitude_confidence`
reports what fraction of an event's own contributions were genuinely
magnitude-scaled vs. direction-only, so nothing pretends to a precision
it doesn't have.

THIS IS A RULE-BASED HYPOTHESIS LAYER, NOT SEMANTIC UNDERSTANDING: every
mapping below is a disclosed, fixed, keyword-triggered rule. It is
genuinely useful and testable (see the validation report's experiments),
but it cannot resolve real ambiguity a human or a real LLM could ("are
these layoffs a sign of distress or of discipline?") - see the
COST_REDUCTION_CUES carve-out below for exactly where that line is drawn,
and llm_schema.py for the designed (not yet active in this environment)
LLM-backed alternative.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from market_agent.research.extraction import AREA_KEYWORDS
from market_agent.research.news_state.epistemic import (
    certainty_for_breakdown, classify_epistemic_status, dominant_status,
)
from market_agent.research.news_state.magnitude import (
    DIRECTION_ONLY_SCORE, MAGNITUDE_CONFIDENCE, explicit_sign, extract_primary_magnitude, magnitude_to_score,
)
from market_agent.research.news_state.schema import EventVector, IMPLICATION_AXES
from market_agent.research.schema import SourceDocument, TimelineEvent

# (trigger areas, target axis, relationship) - a clause matching ANY trigger area contributes to
# `axis`. DIRECT = same sign as the clause's own sentiment; INVERSE = opposite sign. One clause can
# (and often should) populate MULTIPLE axes - see module docstring.
IMPLICATION_RULES: tuple[tuple[frozenset, str, str], ...] = (
    (frozenset({"revenue"}), "growth", "DIRECT"),
    (frozenset({"earnings"}), "profitability", "DIRECT"),
    (frozenset({"guidance"}), "guidance", "DIRECT"),
    (frozenset({"guidance"}), "growth", "DIRECT"),                 # a guidance change is also a weak growth signal
    (frozenset({"cash_flow"}), "cash_flow", "DIRECT"),
    (frozenset({"debt", "capital_allocation"}), "balance_sheet", "DIRECT"),
    (frozenset({"regulatory"}), "regulatory", "DIRECT"),
    (frozenset({"supply_chain"}), "supply_chain", "DIRECT"),
    (frozenset({"competition"}), "competitive_position", "DIRECT"),
    (frozenset({"demand"}), "demand", "DIRECT"),
    # risk is a DERIVED axis, fed by the RISK-RELEVANT reading of several other areas, INVERTED: bad
    # regulatory/geopolitical/workforce/supply-chain/competitive news elevates risk even though it
    # simultaneously depresses the DIRECT axis above (both fire from the same clause, on purpose).
    (frozenset({"regulatory", "geopolitical", "workforce", "supply_chain", "competition"}), "risk", "INVERSE"),
)

# A clause that is negative-sentiment on "workforce" (layoffs) does NOT automatically get a positive
# profitability read - layoffs can equally signal distress, not discipline. Profitability only gets a
# DIRECT-positive contribution when the clause ALSO uses explicit cost-reduction language - a
# narrower, more defensible trigger than "layoffs happened".
COST_REDUCTION_CUES = ("cost-cutting", "cost cutting", "cost reduction", "reduce costs", "reducing costs",
                        "streamline", "cost savings", "lower costs", "cut costs")

MATERIALITY_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.55, "LOW": 0.25}
RELIABILITY_WEIGHT = {"PRIMARY": 1.0, "SECONDARY": 0.6, "TERTIARY": 0.3}
CONFIRMATION_K = 0.7  # fixed, disclosed saturation rate - see aggregation.py's justification

# MODALITY: a distinct dimension from source reliability (RELIABILITY_WEIGHT, epistemic.py's
# CERTAINTY_WEIGHT) - this is about whether the CLAIM ITSELF describes something that happened vs.
# something hedged or hypothetical, independent of who is reporting it. SENTiVENT (see extraction.py's
# NEGATION_CUES docstring for the citation) annotates modality as its own event attribute alongside
# type/subtype/negation for exactly this reason - "analysts expect a 10% decline next quarter" is a
# qualitatively weaker economic claim than "revenue declined 10%", not merely a less-certainly-sourced
# version of the same claim (that distinction already lives in epistemic.py and is NOT duplicated
# here). FACT and REPORTING both describe a claimed past/present occurrence - full weight in the
# implications MEAN. INTERPRETATION and SPECULATION describe an inference or a hedge about the
# future - discounted, not excluded (a real forward-looking claim should still count for something).
# The exact discount factors below are OUR calibration, not a number the cited literature specifies -
# the literature motivates having this dimension at all, not these particular weights.
MODALITY_WEIGHT: dict[str, float] = {"FACT": 1.0, "REPORTING": 1.0, "INTERPRETATION": 0.7, "SPECULATION": 0.4}


def _all_matched_areas(clause: str) -> set[str]:
    """Multi-label version of extraction.py's classify_affected_area -
    returns EVERY area whose keywords appear, not just the first."""
    lower = clause.lower()
    matched = set()
    for area, keywords in AREA_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                matched.add(area)
                break
    return matched


def _sentiment_sign(sentiment: str) -> float | None:
    return {"POSITIVE": 1.0, "NEGATIVE": -1.0}.get(sentiment)  # MIXED/NEUTRAL -> None, no confident direction


def confirmation_strength(independent_source_count: int, avg_reliability: float) -> float:
    """[0,1], SATURATING - diminishing returns per additional confirming
    source. This is the mechanism that lets independent confirmation
    raise CONFIDENCE without the underlying economic-implication value
    itself scaling up - see the validation report's Experiment B."""
    if independent_source_count <= 0:
        return 0.0
    saturation = 1.0 - pow(2.718281828, -CONFIRMATION_K * independent_source_count)
    return round(saturation * avg_reliability, 4)


def _clause_signed_score(event: TimelineEvent) -> tuple[float | None, str]:
    """One clause's signed magnitude score plus a tag for what it came
    from - the fact's own unit ("PERCENT"/"BPS", both anchored and
    scored) or a direction-only fallback ("DIRECTION_ONLY") - matching
    magnitude.py's MAGNITUDE_CONFIDENCE keys exactly, so confidence
    reflects the SPECIFIC kind of evidence used, not a generic bucket."""
    fact = extract_primary_magnitude(event.description)
    lexicon_sign = _sentiment_sign(event.sentiment)

    if fact is not None:
        scaled = magnitude_to_score(fact)  # None for USD (extracted, not scored) - see magnitude.py
        if scaled is not None:
            sign = explicit_sign(fact) or lexicon_sign
            if sign is not None:
                return sign * scaled, fact.unit

    if lexicon_sign is not None:
        return lexicon_sign * DIRECTION_ONLY_SCORE, "DIRECTION_ONLY"
    return None, "DIRECTION_ONLY"


def _implications_from_events(
        events: list[TimelineEvent]) -> tuple[dict[str, list[tuple[float, float]]], list[str], dict[str, set[str]]]:
    """Returns per-axis (value, modality_weight) contributions, the magnitude-confidence source tags,
    and per-axis STATED/INFERRED basis - see IMPLICATION_RULES' DIRECT/INVERSE split and
    COST_REDUCTION_CUES below: a DIRECT rule reports what the clause itself said (STATED); an INVERSE
    rule or the fixed cost-reduction rule derives an axis the clause never named (INFERRED) - "layoffs"
    directly states workforce/demand news but only IMPLIES elevated risk. Surfacing this basis (see
    EventVector.implication_basis) is what lets a reader tell "the article said X" apart from "we
    inferred X from the article saying something else" - the separation this design has always drawn
    internally (DIRECT vs INVERSE), now an explicit, disclosed field instead of only an implementation
    detail."""
    per_axis: dict[str, list[tuple[float, float]]] = {axis: [] for axis in IMPLICATION_AXES}
    basis_by_axis: dict[str, set[str]] = {axis: set() for axis in IMPLICATION_AXES}
    contribution_sources: list[str] = []
    for event in events:
        signed_score, source = _clause_signed_score(event)
        areas = _all_matched_areas(event.description)
        weight = MODALITY_WEIGHT.get(event.evidence_type, 1.0)
        if signed_score is not None:
            fired = False
            for trigger_areas, axis, relationship in IMPLICATION_RULES:
                if areas & trigger_areas:
                    value = signed_score if relationship == "DIRECT" else -signed_score
                    per_axis[axis].append((value, weight))
                    basis_by_axis[axis].add("STATED" if relationship == "DIRECT" else "INFERRED")
                    fired = True
            if fired:
                contribution_sources.append(source)
        if "workforce" in areas and any(cue in event.description.lower() for cue in COST_REDUCTION_CUES):
            per_axis["profitability"].append((1.0, weight))  # fixed, disclosed rule value - not magnitude-scaled
            basis_by_axis["profitability"].add("INFERRED")  # cost-reduction READ, not a stated profitability figure
    return per_axis, contribution_sources, basis_by_axis


def _basis_label(bases: set[str]) -> str | None:
    if not bases:
        return None
    if bases == {"STATED"}:
        return "STATED"
    if bases == {"INFERRED"}:
        return "INFERRED"
    return "MIXED"


def _text_sentiment(events: list[TimelineEvent]) -> float | None:
    signs = [s for s in (_sentiment_sign(e.sentiment) for e in events) if s is not None]
    return round(sum(signs) / len(signs), 4) if signs else None


def _time_horizon(events: list[TimelineEvent]) -> str:
    joined = " ".join(e.description.lower() for e in events)
    if any(p in joined for p in ("next quarter", "this quarter", "near-term", "near term")):
        return "SHORT_TERM"
    if any(p in joined for p in ("this year", "next year", "full-year", "full year", "fiscal year")):
        return "MEDIUM_TERM"
    if any(p in joined for p in ("long-term", "long term", "multi-year", "strategic", "over the coming years")):
        return "LONG_TERM"
    return "UNSPECIFIED"


def build_event_vector(entity: str, cluster: list[TimelineEvent], documents_by_id: dict[str, SourceDocument]
                        ) -> EventVector:
    """`cluster` is ONE real event-instance grouping from
    event_identity.cluster_events - not a whole topic's worth of
    unrelated occurrences."""
    per_axis_raw, contribution_sources, basis_by_axis = _implications_from_events(cluster)
    implications: dict[str, float | None] = {}
    dispersion: dict[str, float | None] = {}
    implication_basis: dict[str, str | None] = {}
    for axis, weighted_vals in per_axis_raw.items():
        implication_basis[axis] = _basis_label(basis_by_axis[axis])
        if not weighted_vals:
            implications[axis], dispersion[axis] = None, None
            continue
        total_w = sum(w for _, w in weighted_vals)
        # MODALITY-WEIGHTED mean: a SPECULATION/INTERPRETATION clause (MODALITY_WEIGHT) still
        # contributes, but counts for less than a stated FACT/REPORTING clause - see MODALITY_WEIGHT's
        # docstring. total_w > 0 always holds here since every weight in MODALITY_WEIGHT is positive.
        mean = sum(v * w for v, w in weighted_vals) / total_w
        implications[axis] = round(mean, 2)  # 2 decimals - matching magnitude.py's own precision; more
        #                                       would be false precision given the underlying anchors
        # WITHIN-event dispersion: modality-weighted variance across this event's own constituent
        # clauses on this axis - a single clause has zero within-event dispersion by definition
        # (nothing to disagree with itself); two opposing clauses correctly register real dispersion
        # here instead of silently vanishing into a 0.0 mean (see schema.py's EventVector.dispersion
        # docstring).
        dispersion[axis] = round(sum(w * (v - mean) ** 2 for v, w in weighted_vals) / total_w, 4)

    magnitude_confidence = (
        round(sum(MAGNITUDE_CONFIDENCE[s] for s in contribution_sources) / len(contribution_sources), 4)
        if contribution_sources else 0.0)

    epistemic_breakdown = dict(Counter(
        classify_epistemic_status(e.description, documents_by_id[e.source_ids[0]].source_type)
        for e in cluster if e.source_ids and e.source_ids[0] in documents_by_id
    ))
    certainty = certainty_for_breakdown(epistemic_breakdown)
    epistemic_status = dominant_status(epistemic_breakdown)

    source_ids = sorted({sid for e in cluster for sid in e.source_ids})
    reliabilities = [RELIABILITY_WEIGHT.get(documents_by_id[sid].reliability, 0.3)
                      for sid in source_ids if sid in documents_by_id]
    source_quality = round(sum(reliabilities) / len(reliabilities), 4) if reliabilities else 0.0

    # independent_source_count: distinct CANONICAL documents behind this cluster - a document already
    # marked as a syndicated duplicate (normalize.py's duplicate_of) does not count again, same
    # discipline as narratives.py's own independent-source counting.
    canonical_ids = {(documents_by_id[sid].duplicate_of or sid) for sid in source_ids if sid in documents_by_id}
    independent_source_count = len(canonical_ids)

    dates = sorted(e.date for e in cluster)
    description = max(cluster, key=lambda e: MATERIALITY_WEIGHT.get(e.materiality, 0.25)).description

    event_vector_id = "EV_" + hashlib.sha256(
        f"{entity}:{sorted(e.event_id for e in cluster)}".encode()).hexdigest()[:12]
    return EventVector(
        event_vector_id=event_vector_id, entity=entity, as_of=dates[-1] if dates else "",
        description=description, implications=implications, text_sentiment=_text_sentiment(cluster),
        materiality=max((MATERIALITY_WEIGHT.get(e.materiality, 0.25) for e in cluster), default=0.25),
        certainty=certainty, epistemic_status=epistemic_status, dispersion=dispersion,
        epistemic_breakdown=epistemic_breakdown, time_horizon=_time_horizon(cluster),
        independent_source_count=independent_source_count, source_quality=source_quality,
        confirmation_strength=confirmation_strength(independent_source_count, source_quality),
        magnitude_confidence=magnitude_confidence, implication_basis=implication_basis,
        constituent_event_ids=[e.event_id for e in cluster], constituent_source_ids=source_ids,
    )


def build_event_vectors(entity: str, clusters: list[list[TimelineEvent]],
                         documents_by_id: dict[str, SourceDocument]) -> list[EventVector]:
    return [build_event_vector(entity, cluster, documents_by_id) for cluster in clusters if cluster]
