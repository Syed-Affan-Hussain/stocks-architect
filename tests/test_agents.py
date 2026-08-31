from datetime import datetime, timezone

import pytest

from market_agent.agents.adaptive_agent import AdaptiveAgent
from market_agent.agents.static_agent import StaticAgent
from market_agent.agents.variants import (
    make_current_adaptive_agent, make_ensemble_adaptive_agent, make_methodology_informed_adaptive_agent,
    make_technical_adaptive_agent,
)
from market_agent.events.schema import EventRecord
from market_agent.store import db

BASELINE = {20: 0.02}  # unconditional |effect| at the 20-day horizon
NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _event(direction="negative", context=None):
    return EventRecord(entity="NVDA", event_type="GUIDANCE_CHANGE", direction=direction, source="wire",
                        source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=NOW,
                        ingested_at=NOW, context=context or {"regime": "NORMAL"})


@pytest.fixture
def conn():
    return db.connect(":memory:")


def test_static_agent_signs_baseline_by_direction():
    agent = StaticAgent(BASELINE)
    neg = agent.predict(_event("negative"), 20, NOW)
    pos = agent.predict(_event("positive"), 20, NOW)
    assert neg.predicted_impact == -0.02
    assert pos.predicted_impact == 0.02


def test_static_agent_insufficient_precedent_for_unknown_horizon():
    agent = StaticAgent(BASELINE)
    result = agent.predict(_event("negative"), horizon_days=5, predicted_at=NOW)
    assert result.predicted_confidence == "INSUFFICIENT_PRECEDENT"
    assert result.predicted_impact is None


def test_adaptive_agent_matches_static_when_no_relationship_exists(conn):
    static = StaticAgent(BASELINE)
    adaptive = AdaptiveAgent(conn, BASELINE)
    event = _event("negative")
    assert adaptive.predict(event, 20, NOW).predicted_impact == static.predict(event, 20, NOW).predicted_impact


def test_adaptive_agent_uses_validated_relationship_when_it_matches(conn):
    db.upsert_relationship(conn, "rel-1", condition={"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                       "regime": "RISK_OFF"},
                            horizon_days=20, effect_estimate=-0.09, ci_low=None, ci_high=None, n_supporting=60,
                            status="ACTIVE", created_at=NOW)
    adaptive = AdaptiveAgent(conn, BASELINE)
    result = adaptive.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert result.predicted_impact == -0.09
    assert result.basis["relationship_id"] == "rel-1"
    assert result.predicted_confidence == "HIGH"  # n_supporting=60 clears the HIGH threshold (>=50)


def test_adaptive_agent_ignores_relationship_for_a_different_regime(conn):
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 40, "ACTIVE", NOW)
    adaptive = AdaptiveAgent(conn, BASELINE)
    result = adaptive.predict(_event("negative", context={"regime": "NORMAL"}), 20, NOW)
    assert result.basis["basis"] == "unconditional_baseline"  # fell back - the RISK_OFF relationship doesn't apply


def test_adaptive_agent_ignores_retired_relationship(conn):
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 40, "RETIRED", NOW)
    adaptive = AdaptiveAgent(conn, BASELINE)
    result = adaptive.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert result.basis["basis"] == "unconditional_baseline"


def test_static_agent_is_frozen_even_after_adaptive_promotes_a_relationship(conn):
    """The core Static-vs-Adaptive guarantee: promoting a relationship must
    never change what Agent-STATIC predicts for the same future event."""
    static = StaticAgent(BASELINE)
    before = static.predict(_event("negative"), 20, NOW).predicted_impact

    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "NORMAL"}, 20, -0.30, None, None, 100, "ACTIVE", NOW)

    after = static.predict(_event("negative"), 20, NOW).predicted_impact
    assert before == after == -0.02  # completely unaffected by what was promoted


# --- stage 6: concept_filter and the three named ADAPTIVE variants ---

def _seed_event_context_relationship(conn):
    db.upsert_relationship(conn, "rel-event-context", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                          "regime": "RISK_OFF"},
                            20, -0.09, None, None, 40, "ACTIVE", NOW)  # no concept - pure event-context


def _seed_technical_relationship(conn, methodology_ids=None):
    db.upsert_relationship(conn, "rel-technical", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                     "breakout_state": "BREAKOUT_DOWN"},
                            20, -0.12, None, None, 40, "ACTIVE", NOW, concept="BREAKOUT",
                            methodology_ids=methodology_ids)


def test_unrestricted_concept_filter_default_matches_both_kinds(conn):
    _seed_event_context_relationship(conn)
    _seed_technical_relationship(conn)
    adaptive = AdaptiveAgent(conn, BASELINE)  # default concept_filter=None - pre-stage-6 behavior
    r1 = adaptive.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    r2 = adaptive.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert r1.basis["relationship_id"] == "rel-event-context"
    assert r2.basis["relationship_id"] == "rel-technical"


def test_invalid_concept_filter_raises():
    with pytest.raises(ValueError):
        AdaptiveAgent(db.connect(":memory:"), BASELINE, concept_filter="NOT_A_REAL_FILTER")


def test_current_adaptive_only_matches_event_context_relationships(conn):
    _seed_event_context_relationship(conn)
    _seed_technical_relationship(conn)
    agent = make_current_adaptive_agent(conn, BASELINE)

    matched = agent.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert matched.basis["relationship_id"] == "rel-event-context"

    ignored = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert ignored.basis["basis"] == "unconditional_baseline"  # technical relationship invisible to this agent
    assert agent.model_version == "CURRENT_ADAPTIVE_v1"


def test_technical_adaptive_only_matches_concept_relationships_regardless_of_methodology(conn):
    _seed_event_context_relationship(conn)
    _seed_technical_relationship(conn, methodology_ids=None)  # no methodology backing at all
    agent = make_technical_adaptive_agent(conn, BASELINE)

    matched = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert matched.basis["relationship_id"] == "rel-technical"

    ignored = agent.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert ignored.basis["basis"] == "unconditional_baseline"  # event-context-only relationship invisible here
    assert agent.model_version == "TECHNICAL_ADAPTIVE_v1"


def test_methodology_informed_adaptive_requires_methodology_backing(conn):
    _seed_technical_relationship(conn, methodology_ids=None)  # concept present, but NO methodology backing
    agent = make_methodology_informed_adaptive_agent(conn, BASELINE)
    result = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert result.basis["basis"] == "unconditional_baseline"  # not methodology-backed, so ineligible
    assert agent.model_version == "METHODOLOGY_ADAPTIVE_v1"


def test_methodology_informed_adaptive_matches_when_methodology_backed(conn):
    _seed_technical_relationship(conn, methodology_ids=["meth-1"])
    agent = make_methodology_informed_adaptive_agent(conn, BASELINE)
    result = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert result.basis["relationship_id"] == "rel-technical"


# --- stage 7: ENSEMBLE_ADAPTIVE - governed selection via qualified_relationship_ids ---

def test_qualified_relationship_ids_restricts_to_only_that_set(conn):
    db.upsert_relationship(conn, "rel-a", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.09, None, None, 40,
                            "ACTIVE", NOW, concept="BREAKOUT")
    db.upsert_relationship(conn, "rel-b", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.12, None, None, 40, "ACTIVE", NOW)
    agent = AdaptiveAgent(conn, BASELINE, qualified_relationship_ids={"rel-a"})

    matched = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert matched.basis["relationship_id"] == "rel-a"

    not_qualified = agent.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert not_qualified.basis["basis"] == "unconditional_baseline"  # rel-b is ACTIVE but NOT qualified


def test_empty_qualified_set_means_nothing_is_eligible(conn):
    db.upsert_relationship(conn, "rel-a", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "regime": "RISK_OFF"}, 20, -0.09, None, None, 40, "ACTIVE", NOW)
    agent = AdaptiveAgent(conn, BASELINE, qualified_relationship_ids=set())
    result = agent.predict(_event("negative", context={"regime": "RISK_OFF"}), 20, NOW)
    assert result.basis["basis"] == "unconditional_baseline"


def test_ensemble_adaptive_agent_is_a_selection_not_a_blend(conn):
    """The ensemble's prediction for a matching, qualified relationship
    must be EXACTLY that relationship's own effect_estimate - never an
    average with the unconditional baseline or any other component."""
    db.upsert_relationship(conn, "rel-a", {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                            "breakout_state": "BREAKOUT_DOWN"}, 20, -0.15, None, None, 40,
                            "ACTIVE", NOW, concept="BREAKOUT")
    agent = make_ensemble_adaptive_agent(conn, BASELINE, qualified_relationship_ids={"rel-a"})
    result = agent.predict(_event("negative", context={"breakout_state": "BREAKOUT_DOWN"}), 20, NOW)
    assert result.predicted_impact == -0.15  # exactly rel-a's own effect_estimate, not blended with anything
    assert agent.model_version == "ENSEMBLE_ADAPTIVE_v1"
