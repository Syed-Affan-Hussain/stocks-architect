from datetime import datetime, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.hypothesis import (
    MAX_CONDITIONING_VARS, MAX_TECHNICAL_DIMENSIONS_PER_EVENT, RuleBasedHypothesisGenerator, formalize_and_store,
)
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _row(conn, regime="RISK_OFF", prior_5d_return=None, realized_vol_20d=None, technical=None):
    context = {} if regime is None else {"regime": regime}
    if prior_5d_return is not None:
        context["prior_5d_return"] = prior_5d_return
    if realized_vol_20d is not None:
        context["realized_vol_20d"] = realized_vol_20d
    if technical:
        context.update(technical)
    event = EventRecord(entity="NVDA", event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=NOW,
                         ingested_at=NOW, context=context)
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "TEST_v1", NOW)
    event_id = db.log_prediction(conn, event, pred)
    return db.get_event(conn, event_id)


def test_single_available_dimension_proposes_exactly_one_hypothesis():
    """Only regime available -> the powerset of {regime} is just {regime} itself."""
    conn = db.connect(":memory:")
    row = _row(conn)
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 1
    assert proposed[0].condition == {"event_type": "GUIDANCE_CHANGE", "direction": "negative", "regime": "RISK_OFF"}
    assert "regime=" in proposed[0].explanation_text


def test_two_available_dimensions_proposes_the_full_powerset():
    """regime + prior_return_bucket available -> 2^2-1 = 3 candidates: each alone, and combined."""
    conn = db.connect(":memory:")
    row = _row(conn, prior_5d_return=-0.09)  # "LARGE_DECLINE" bucket
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 3
    conditions = [p.condition for p in proposed]
    base = {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}
    assert {**base, "regime": "RISK_OFF"} in conditions
    assert {**base, "prior_return_bucket": "LARGE_DECLINE"} in conditions
    assert {**base, "regime": "RISK_OFF", "prior_return_bucket": "LARGE_DECLINE"} in conditions
    combined = next(p for p in proposed if len(p.condition) == 4)
    assert "regime=" in combined.explanation_text and "prior_return_bucket=" in combined.explanation_text


def test_three_available_dimensions_respects_the_complexity_cap():
    """All 3 dimensions available -> 2^3-1 = 7 candidates, none exceeding MAX_CONDITIONING_VARS
    conditioning variables (i.e. at most MAX_CONDITIONING_VARS + 2 total condition keys, counting
    the mandatory event_type/direction)."""
    conn = db.connect(":memory:")
    row = _row(conn, prior_5d_return=-0.09, realized_vol_20d=0.05)  # LARGE_DECLINE, HIGH_VOL
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 7
    for p in proposed:
        n_conditioning_vars = len(p.condition) - 2  # minus event_type, direction
        assert 1 <= n_conditioning_vars <= MAX_CONDITIONING_VARS
    # every candidate condition is genuinely distinct
    seen = [frozenset(p.condition.items()) for p in proposed]
    assert len(seen) == len(set(seen))


def test_unknown_dimensions_are_excluded_not_matched_as_a_value():
    """A missing prior_5d_return must not silently participate as
    prior_return_bucket='UNKNOWN' - that would falsely cluster genuinely
    different unknown-context events together."""
    conn = db.connect(":memory:")
    row = _row(conn)  # no prior_5d_return, no realized_vol_20d
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    for p in proposed:
        assert "UNKNOWN" not in p.condition.values()


def test_generator_refuses_non_learnable_error_types():
    conn = db.connect(":memory:")
    row = _row(conn)
    assert RuleBasedHypothesisGenerator().generate(row, "NOVEL_EVENT") == []
    assert RuleBasedHypothesisGenerator().generate(row, "CONFOUNDING_EVENT") == []
    assert RuleBasedHypothesisGenerator().generate(row, "DATA_ERROR") == []


def test_formalize_and_store_writes_a_candidate_hypothesis_per_proposal():
    conn = db.connect(":memory:")
    row = _row(conn, prior_5d_return=-0.09)
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), row, "WRONG_DIRECTION", 20, NOW)
    assert len(hids) == 3
    assert len(db.untested_hypotheses(conn)) == 3


def test_formalize_and_store_returns_empty_list_for_non_learnable_error():
    conn = db.connect(":memory:")
    row = _row(conn)
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), row, "OK", 20, NOW)
    assert hids == []
    assert db.untested_hypotheses(conn) == []


# --- stage 6: technical trading-concept dimensions ---

def test_single_technical_dimension_available_proposes_a_hypothesis_over_it():
    conn = db.connect(":memory:")
    row = _row(conn, regime=None, technical={"breakout_state": "BREAKOUT_UP"})
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 1
    assert proposed[0].condition["breakout_state"] == "BREAKOUT_UP"


def test_technical_dimension_combines_with_event_context_dimension():
    """'Bounded combinations of trading concepts AND interactions with
    event context' - a technical dimension can pair with regime through
    the same powerset mechanism, no special-casing needed."""
    conn = db.connect(":memory:")
    row = _row(conn, regime="RISK_OFF", technical={"breakout_state": "BREAKOUT_UP"})
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    conditions = [p.condition for p in proposed]
    assert len(proposed) == 3  # regime alone, breakout_state alone, both combined
    combined = next(c for c in conditions if len(c) == 4)
    assert combined["regime"] == "RISK_OFF" and combined["breakout_state"] == "BREAKOUT_UP"


def test_technical_dimensions_with_default_values_are_excluded():
    """FLAT trend / NONE breakout / NORMAL volatility are the 'nothing
    interesting happening' states - same exclusion discipline as UNKNOWN."""
    conn = db.connect(":memory:")
    row = _row(conn, technical={"trend_direction": "FLAT", "breakout_state": "NONE",
                                 "volatility_state": "NORMAL"})
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 1  # only regime survives
    assert proposed[0].condition == {"event_type": "GUIDANCE_CHANGE", "direction": "negative", "regime": "RISK_OFF"}


def test_technical_dimension_pool_is_bounded_even_with_many_interesting_dimensions():
    """Every one of the 15 technical dimensions set to an 'interesting'
    value at once must still only ever contribute
    MAX_TECHNICAL_DIMENSIONS_PER_EVENT of them to the pool - proving the
    pool-size bound actually holds, not just the arity cap."""
    conn = db.connect(":memory:")
    all_interesting = {
        "trend_direction": "UP", "momentum_state": "POSITIVE", "breakout_state": "BREAKOUT_UP",
        "mean_reversion_state": "OVEREXTENDED_HIGH", "pullback_state": "PULLBACK_IN_UPTREND",
        "volatility_state": "EXPANSION", "relative_volume_state": "HIGH_RVOL",
        "relative_strength_state": "OUTPERFORMING", "ma_stack": "BULLISH_STACK",
        "market_structure": "HIGHER_HIGHS_HIGHER_LOWS", "gap_state": "GAP_UP",
        "vwap_state": "ABOVE_VWAP_PROXY", "support_resistance_state": "NEAR_RESISTANCE",
        "price_action_pattern": "BULLISH_ENGULFING", "mtf_confirmation": "CONFIRMED",
    }
    row = _row(conn, regime=None, technical=all_interesting)  # no regime/prior_return/vol_bucket set
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    technical_dims_seen = set()
    for p in proposed:
        for key in p.condition:
            if key in all_interesting:
                technical_dims_seen.add(key)
    assert len(technical_dims_seen) == MAX_TECHNICAL_DIMENSIONS_PER_EVENT
    # pool size 3 -> at most C(3,1)+C(3,2)+C(3,3) = 7 candidates
    assert len(proposed) == 7


def test_formalize_and_store_records_concept_when_technical_dimension_present():
    conn = db.connect(":memory:")
    row = _row(conn, regime=None, technical={"breakout_state": "BREAKOUT_UP"})
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), row, "WRONG_DIRECTION", 20, NOW)
    assert len(hids) == 1
    stored = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hids[0],)).fetchone()
    assert stored["concept"] == "BREAKOUT"
    assert stored["methodology_ids_json"] is None  # no methodology ingested yet in this test


def test_formalize_and_store_records_no_concept_for_pure_event_context_hypothesis():
    conn = db.connect(":memory:")
    row = _row(conn)  # regime only, no technical dimensions
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), row, "WRONG_DIRECTION", 20, NOW)
    stored = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hids[0],)).fetchone()
    assert stored["concept"] is None


def test_include_technical_dimensions_false_reproduces_stage5_event_context_only_behavior():
    """Stage 7: a generator constructed with include_technical_dimensions=False
    must behave exactly like stage 5's generator - technical dimensions
    present in context are ignored entirely, even a single one."""
    conn = db.connect(":memory:")
    row = _row(conn, technical={"breakout_state": "BREAKOUT_UP"})  # regime="RISK_OFF" default + technical
    proposed = RuleBasedHypothesisGenerator(include_technical_dimensions=False).generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 1
    assert proposed[0].condition == {"event_type": "GUIDANCE_CHANGE", "direction": "negative", "regime": "RISK_OFF"}
    assert "breakout_state" not in proposed[0].condition


def test_include_technical_dimensions_defaults_to_true_for_backward_compatibility():
    conn = db.connect(":memory:")
    row = _row(conn, regime=None, technical={"breakout_state": "BREAKOUT_UP"})
    proposed = RuleBasedHypothesisGenerator().generate(row, "WRONG_DIRECTION")
    assert len(proposed) == 1
    assert proposed[0].condition["breakout_state"] == "BREAKOUT_UP"


def test_formalize_and_store_links_methodology_ids_for_a_seeded_concept():
    from market_agent.methodology.extractor import RuleBasedMethodologyExtractor
    from market_agent.methodology.ingest import ingest_methodology
    from market_agent.methodology.schema import RawMethodologySource

    conn = db.connect(":memory:")
    source = RawMethodologySource(name="Test System", practitioner="Test Trader", source_type="book",
                                   raw_text="A breakout system.")
    methodology_id = ingest_methodology(conn, RuleBasedMethodologyExtractor(), source, NOW)

    row = _row(conn, regime=None, technical={"breakout_state": "BREAKOUT_UP"})
    hids = formalize_and_store(conn, RuleBasedHypothesisGenerator(), row, "WRONG_DIRECTION", 20, NOW)
    stored = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hids[0],)).fetchone()
    import json
    assert json.loads(stored["methodology_ids_json"]) == [methodology_id]
