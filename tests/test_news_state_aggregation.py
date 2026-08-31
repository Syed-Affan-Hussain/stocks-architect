import json
from datetime import datetime, timedelta, timezone

from market_agent.research.news_state.aggregation import (
    aggregate_company_state, compute_novelty, event_weight, persist_event_vectors,
)
from market_agent.research.news_state.schema import CompanyNewsState, EventVector, IMPLICATION_AXES
from market_agent.store import db

NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)


def _implications(**overrides):
    d = {axis: None for axis in IMPLICATION_AXES}
    d.update(overrides)
    return d


def _event(event_vector_id, as_of, materiality=1.0, confirmation=1.0, certainty=0.6, source_quality=0.6,
           text_sentiment=None, **implication_overrides):
    return EventVector(event_vector_id=event_vector_id, entity="ACME", as_of=as_of, description="d",
                        implications=_implications(**implication_overrides), text_sentiment=text_sentiment,
                        materiality=materiality, certainty=certainty, epistemic_status="THIRD_PARTY_REPORTING",
                        confirmation_strength=confirmation, source_quality=source_quality,
                        independent_source_count=1)


def test_event_weight_decays_with_age():
    fresh = _event("e1", NOW.isoformat())
    old = _event("e2", (NOW - timedelta(days=14)).isoformat())
    assert event_weight(fresh, NOW, half_life_days=7.0) > event_weight(old, NOW, half_life_days=7.0)


def test_event_weight_zero_age_equals_no_decay():
    e = _event("e1", NOW.isoformat())
    w = event_weight(e, NOW, half_life_days=7.0)
    assert abs(w - (1.0 * e.materiality * (0.3 + 0.7 * e.confirmation_strength))) < 1e-9


def test_aggregate_state_weighted_mean_of_a_single_axis():
    e1 = _event("e1", NOW.isoformat(), growth=1.0)
    e2 = _event("e2", NOW.isoformat(), growth=1.0)
    state = aggregate_company_state("ACME", [e1, e2], NOW, raw_document_count=2)
    assert state.dimensions["growth"] == 1.0
    assert state.dispersion["growth"] == 0.0
    assert "growth" not in state.contradiction_axes


def test_direction_only_magnitude_contradiction_is_still_flagged():
    """The common real-world case after magnitude-aware scoring: two
    clauses with no extractable number, opposed in direction, each
    scored at +-0.5 (DIRECTION_ONLY_SCORE), not +-1.0. This must still
    cross the contradiction threshold - it was the reason the threshold
    was recalibrated down from 0.4 to 0.15."""
    positive = _event("e1", NOW.isoformat(), demand=0.5)
    negative = _event("e2", NOW.isoformat(), demand=-0.5)
    state = aggregate_company_state("ACME", [positive, negative], NOW, raw_document_count=2)
    assert state.dimensions["demand"] == 0.0
    assert state.dispersion["demand"] == 0.25
    assert "demand" in state.contradiction_axes


def test_contradictory_events_move_mean_toward_neutral_and_raise_dispersion():
    positive = _event("e1", NOW.isoformat(), demand=1.0)
    negative = _event("e2", NOW.isoformat(), demand=-1.0)
    state = aggregate_company_state("ACME", [positive, negative], NOW, raw_document_count=2)
    assert state.dimensions["demand"] == 0.0
    assert state.dispersion["demand"] > 0
    assert "demand" in state.contradiction_axes


def test_within_event_dispersion_is_not_lost_when_events_are_pre_averaged():
    """A single EventVector whose OWN constituent clauses already
    disagreed on an axis (event_vector.py averaged them to a neutral
    mean) must still contribute its real internal dispersion to the
    company-level total - the law-of-total-variance combination in
    _weighted_axis, not just the trivially-zero variance of a single
    already-averaged data point."""
    contradictory_within_one_event = _event("e1", NOW.isoformat(), demand=0.0)
    contradictory_within_one_event.dispersion["demand"] = 1.0  # this ONE event already disagreed internally
    state = aggregate_company_state("ACME", [contradictory_within_one_event], NOW, raw_document_count=1)
    assert state.dimensions["demand"] == 0.0
    assert state.dispersion["demand"] == 1.0  # NOT 0.0 - the within-event disagreement must survive
    assert "demand" in state.contradiction_axes


def test_between_and_within_event_dispersion_combine_additively():
    e1 = _event("e1", NOW.isoformat(), demand=1.0)   # no internal disagreement
    e2 = _event("e2", NOW.isoformat(), demand=-1.0)  # no internal disagreement
    state = aggregate_company_state("ACME", [e1, e2], NOW, raw_document_count=2)
    assert state.dimensions["demand"] == 0.0
    assert state.dispersion["demand"] == 1.0  # pure between-event variance, matches the hand-computed value


def test_axis_with_no_contributing_events_stays_none():
    e1 = _event("e1", NOW.isoformat(), growth=1.0)
    state = aggregate_company_state("ACME", [e1], NOW, raw_document_count=1)
    assert state.dimensions["risk"] is None
    assert state.dispersion["risk"] is None


def test_novelty_none_below_min_baseline():
    conn = db.connect(":memory:")
    e = _event("e_new", NOW.isoformat(), growth=1.0)
    assert compute_novelty(conn, e) is None  # no prior events at all


def test_novelty_computed_once_enough_prior_events_exist():
    conn = db.connect(":memory:")
    prior_events = [_event(f"e{i}", (NOW - timedelta(days=10 + i)).isoformat(), growth=1.0) for i in range(3)]
    persist_event_vectors(conn, prior_events, NOW - timedelta(days=5))
    similar = _event("e_new", NOW.isoformat(), growth=0.9)
    far = _event("e_far", NOW.isoformat(), growth=-1.0, risk=1.0)
    assert compute_novelty(conn, similar) is not None
    assert compute_novelty(conn, similar) < compute_novelty(conn, far)  # more different -> more novel


def test_state_change_velocity_and_direction_computed_against_prior_state():
    prior = CompanyNewsState(entity="ACME", as_of=(NOW - timedelta(days=1)).isoformat(),
                              dimensions=_implications(growth=0.2), dispersion=_implications())
    e1 = _event("e1", NOW.isoformat(), growth=0.8)
    state = aggregate_company_state("ACME", [e1], NOW, raw_document_count=1, prior_state=prior)
    assert state.state_change["growth"] == 0.6
    assert state.state_velocity is not None and state.state_velocity > 0
    assert state.state_direction["growth"] == 1.0  # only one non-null delta axis - full unit magnitude


def test_no_prior_state_means_no_trajectory_fields():
    e1 = _event("e1", NOW.isoformat(), growth=0.8)
    state = aggregate_company_state("ACME", [e1], NOW, raw_document_count=1)
    assert state.state_change is None
    assert state.state_velocity is None
