"""Aggregates many EventVectors into ONE CompanyNewsState N(t) - the
Layer C object (schema.py). This is where the duplicate-resistance and
contradiction-representation requirements actually get enforced
mathematically, not just described.

WEIGHTING IS EVENT-LEVEL, NEVER ARTICLE-LEVEL: by the time anything
reaches this module, syndicated duplicates have already collapsed into
one EventVector (normalize.py + narratives.py, reused by event_vector.py)
- there is no "article count" anywhere in the weight formula below. Adding
20 more syndicated copies of an already-seen event does not create new
EventVectors, so it cannot inflate N(t) - see the validation report's
Experiment A for the measured ΔN.

DECAY HALF-LIFE IS A DISCLOSED, FIXED CHOICE, NOT A FITTED ONE:
DEFAULT_HALF_LIFE_DAYS=7.0 approximates a typical company-news attention
cycle. There is no labeled dataset in this project of "when did the
market/media actually stop caring about this story" to fit a real decay
rate against - fitting one would require exactly the kind of
outcome-labeled panel this project does not yet have (same gap as the
state-vector report's PCA discussion). Treat 7 days as a stated
assumption, changeable as a disclosed, versioned decision.

DISPERSION IS KEPT, NOT DISCARDED: when contributing events disagree on
an axis, the weighted mean moves toward neutral - a real, correct
mathematical consequence of averaging opposite-signed values - AND the
weighted VARIANCE is reported alongside it, flagging the axis as
`contradiction_axes` above a fixed, disclosed threshold. Both things are
true at once; picking only one would hide information the other supplies
(see the validation report's Experiment C for the measured behavior).
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone

from market_agent.research.news_state.schema import CompanyNewsState, EventVector, IMPLICATION_AXES
from market_agent.store import db

DEFAULT_HALF_LIFE_DAYS = 7.0
# RECALIBRATED after magnitude-aware scoring replaced the flat +-1.0 per clause (event_vector.py):
# most real clauses have no extractable magnitude and fall back to DIRECTION_ONLY_SCORE=0.5, not 1.0
# (magnitude.py) - a clean, maximally-opposed +0.5/-0.5 split now produces dispersion=0.25, not the
# 1.0 a +1.0/-1.0 split under the old flat scoring would have. The threshold was originally set
# assuming +-1.0 was the common case; left at 0.4 it would silently fail to flag the single most
# common real contradiction pattern (two direction-only clauses that flatly disagree). Lowered to
# 0.15 - below the 0.25 ceiling of the common direction-only case, so a genuine direction-only
# disagreement is always caught, while a single event's own (zero, by definition) internal dispersion
# can never trigger it by accident.
CONTRADICTION_VARIANCE_THRESHOLD = 0.15
CONFIRMATION_FLOOR = 0.3  # a single-source event still counts at 30% weight, never zeroed out just
#                            for lacking independent confirmation - confirmation SCALES weight, it
#                            does not gate it entirely (a real, single-source event is still real)
MIN_NOVELTY_BASELINE = 3  # fewer than this many prior events for the entity -> novelty stays None,
#                            never computed from too small a baseline to mean anything


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def event_weight(event: EventVector, as_of: datetime, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """decay(age) x materiality x confirmation - the ONLY three inputs to
    an event's aggregation weight. No term here is "number of articles"."""
    try:
        age_days = max((as_of - _parse_iso(event.as_of)).total_seconds() / 86400.0, 0.0)
    except (ValueError, TypeError):
        age_days = 0.0
    decay = 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0
    confirmation = CONFIRMATION_FLOOR + (1 - CONFIRMATION_FLOOR) * event.confirmation_strength
    return decay * event.materiality * confirmation


def _weighted_axis(events: list[EventVector], axis: str, weights: list[float]) -> tuple[float | None, float | None]:
    """LAW OF TOTAL VARIANCE: Var(total) = E[Var(within groups)] + Var(E[between groups]). An
    EventVector's own `dispersion` (event_vector.py) already captures disagreement AMONG that one
    event's constituent clauses; this function ALSO captures disagreement BETWEEN separate
    EventVectors, and correctly combines both rather than reporting only whichever one happens to be
    computed last. Found necessary by a real failing experiment: two opposing clauses that clustered
    into the SAME narrative (and therefore the same EventVector) were silently averaged to a neutral
    0.0 with zero reported dispersion, because the old between-events-only variance had nothing left
    to disagree about once the within-event averaging had already erased the disagreement."""
    contributing = [(e.implications[axis], e.dispersion.get(axis) or 0.0, w) for e, w in zip(events, weights)
                     if e.implications.get(axis) is not None]
    if not contributing:
        return None, None
    total_w = sum(w for _, _, w in contributing)
    if total_w <= 0:
        return None, None
    mean = sum(v * w for v, _, w in contributing) / total_w
    between_variance = sum(w * (v - mean) ** 2 for v, _, w in contributing) / total_w
    within_variance = sum(w * d for _, d, w in contributing) / total_w
    return round(mean, 4), round(between_variance + within_variance, 4)


def compute_novelty(conn: sqlite3.Connection, event: EventVector) -> float | None:
    """Euclidean distance (over shared, non-null axes) from the NEAREST
    prior real event for this SAME entity - never a fabricated baseline.
    Returns None if fewer than MIN_NOVELTY_BASELINE prior events exist for
    this entity (this pipeline has never seen enough of this company's
    history yet - a disclosed gap, not a guess)."""
    prior_rows = db.prior_news_event_vectors(conn, event.entity, event.as_of)
    if len(prior_rows) < MIN_NOVELTY_BASELINE:
        return None
    import json
    min_distance = None
    for row in prior_rows:
        prior = json.loads(row["event_vector_json"])
        prior_implications = prior.get("implications", {})
        shared = [(event.implications[a], prior_implications.get(a)) for a in IMPLICATION_AXES
                  if event.implications.get(a) is not None and prior_implications.get(a) is not None]
        if not shared:
            continue
        distance = math.sqrt(sum((a - b) ** 2 for a, b in shared) / len(shared))
        if min_distance is None or distance < min_distance:
            min_distance = distance
    return round(min_distance, 4) if min_distance is not None else None


def aggregate_company_state(entity: str, events: list[EventVector], as_of: datetime,
                             raw_document_count: int, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                             prior_state: CompanyNewsState | None = None) -> CompanyNewsState:
    weights = [event_weight(e, as_of, half_life_days) for e in events]
    dimensions: dict[str, float | None] = {}
    dispersion: dict[str, float | None] = {}
    for axis in IMPLICATION_AXES:
        mean, variance = _weighted_axis(events, axis, weights)
        dimensions[axis] = mean
        dispersion[axis] = variance
    contradiction_axes = [a for a, v in dispersion.items() if v is not None and v > CONTRADICTION_VARIANCE_THRESHOLD]

    sentiments = [(e.text_sentiment, w) for e, w in zip(events, weights) if e.text_sentiment is not None]
    text_sentiment = (round(sum(s * w for s, w in sentiments) / sum(w for _, w in sentiments), 4)
                       if sentiments and sum(w for _, w in sentiments) > 0 else None)

    total_w = sum(weights) or 1.0
    # confidence blends EPISTEMIC certainty with CORROBORATION (confirmation_strength) at the
    # per-event level, not just as an aggregation weight across MULTIPLE events - with only one
    # contributing EventVector (the common case: one real story, confirmed by several independent
    # sources), a weighted average of ONE value ignores its own weight entirely (found via a failing
    # experiment: confidence did not move at all when 1 source became 5, because averaging one number
    # with itself returns that number regardless of weight). Corroboration must raise the event's OWN
    # reported confidence, not just its influence relative to OTHER events.
    event_confidences = [e.certainty * (0.5 + 0.5 * e.confirmation_strength) for e in events]
    confidence = (round(sum(c * w for c, w in zip(event_confidences, weights)) / total_w, 4)
                  if events else 0.0)
    source_quality = round(sum(e.source_quality * w for e, w in zip(events, weights)) / total_w, 4) if events else 0.0

    dominant = sorted(events, key=lambda e: -e.materiality)[:5]

    state = CompanyNewsState(
        entity=entity, as_of=as_of.isoformat(), dimensions=dimensions, dispersion=dispersion,
        text_sentiment=text_sentiment, confidence=confidence, news_volume=raw_document_count,
        independent_event_count=len(events), dominant_event_ids=[e.event_vector_id for e in dominant],
        contradiction_axes=contradiction_axes, source_quality=source_quality, half_life_days=half_life_days,
    )

    if prior_state is not None:
        delta: dict[str, float] = {}
        for axis in IMPLICATION_AXES:
            cur, prev = dimensions.get(axis), prior_state.dimensions.get(axis)
            if cur is not None and prev is not None:
                delta[axis] = round(cur - prev, 4)
        state.state_change = delta
        if delta:
            magnitude = math.sqrt(sum(v ** 2 for v in delta.values()))
            elapsed_days = max((as_of - _parse_iso(prior_state.as_of)).total_seconds() / 86400.0, 1e-6)
            state.state_velocity = round(magnitude / elapsed_days, 6)
            state.state_direction = ({k: round(v / magnitude, 4) for k, v in delta.items()} if magnitude > 0 else
                                      {k: 0.0 for k in delta})
    return state


def persist_event_vectors(conn: sqlite3.Connection, events: list[EventVector], computed_at: datetime) -> None:
    for e in events:
        db.save_news_event_vector(conn, e.event_vector_id, e.entity, e.as_of, computed_at, e.to_dict())


def persist_company_state(conn: sqlite3.Connection, state: CompanyNewsState, computed_at: datetime) -> str:
    return db.save_news_company_state(conn, state.entity, state.as_of, computed_at, state.to_dict())
