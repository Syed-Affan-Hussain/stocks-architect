from datetime import datetime, timedelta, timezone

import pytest

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.hierarchical_research import DEFAULT_RESEARCH_BUDGET, FamilyScreeningResult, HierarchicalResearchReport, LevelTestResult
from market_agent.learn.hypothesis_testing import HypothesisTestResult
from market_agent.learn.incremental_value import IncrementalValueResult
from market_agent.learn.overfitting_diagnostics import PermutationTestResult, TemporalStabilityResult
from market_agent.outcomes.ohlcv import Bar, OHLCVProvider
from market_agent.reporting.stage7_final_report import (
    EVIDENCE_STATES, RelationshipTrajectory, answer_final_report_questions, build_stage7_final_report,
    evaluate_relationship_trajectory,
)
from market_agent.store import db
from market_agent.strategy.test_isolation import TestIsolationGuard, TestIsolationViolation

BASELINE = {20: 0.02}
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
TEST_BOUNDARY = (START + timedelta(days=100)).isoformat()
CONDITION = {"event_type": "GUIDANCE_CHANGE", "direction": "positive", "breakout_state": "BREAKOUT_UP"}


class FakeOHLCV(OHLCVProvider):
    def __init__(self):
        self.data: dict[str, list[Bar]] = {}

    def set_bars(self, ticker, bars):
        self.data[ticker] = bars

    def bars(self, ticker, as_of, lookback_days):
        rows = self.data.get(ticker, [])
        cutoff = as_of - timedelta(days=lookback_days)
        return [b for b in rows if b.date <= as_of and b.date >= cutoff]


def _rising_bars(start, n_days=25, daily_pct=0.0025, start_price=100.0):
    bars, price = [], start_price
    for i in range(n_days):
        o = price
        price = price * (1 + daily_pct)
        c = price
        h, l = max(o, c) * 1.001, min(o, c) * 0.999
        bars.append(Bar(date=start + timedelta(days=i), open=o, high=h, low=l, close=c, volume=1_000_000))
    return bars


def _seed_event(conn, entity, published, context, horizon_days=20):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                         source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                         ingested_at=published, context=context)
    pred = PredictionRecord(horizon_days, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, 0.05, published + timedelta(days=horizon_days), 0.03, "OK")


def _passing_level_result(condition=None, n=40, effect=0.05, permutation_status="SURVIVES_PERMUTATION",
                           stability_status="STABLE_ACROSS_TIME", incremental_status="INCREMENTAL_VALUE_CONFIRMED",
                           test_status="CONFIRMED") -> LevelTestResult:
    condition = condition or CONDITION
    test_result = HypothesisTestResult("hid-1", test_status, n, effect, 0.02, 0.001, 0.001, ci_low=0.01, ci_high=0.09)
    incremental = IncrementalValueResult("label", n, 0.03, 0.001, 0.01, 0.05, incremental_status)
    permutation = PermutationTestResult(n, effect, 100, 2000, 0.01, permutation_status)
    stability = TemporalStabilityResult(n, n // 2, n - n // 2, effect, effect, True, stability_status)
    return LevelTestResult(test_result=test_result, incremental_value=incremental, permutation_test=permutation,
                            temporal_stability=stability, condition=condition)


def _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=15, validate_up=True, test_up=True):
    """Seeds a real, ACTIVE-eligible relationship plus real matching
    episodic_events + real rising/falling OHLCV bars for both VALIDATE and
    TEST segments - everything evaluate_relationship_trajectory needs to
    actually build real trades, not synthetic ones."""
    db.upsert_relationship(conn, "rel-econ", CONDITION, 20, 0.05, 0.01, 0.09, 40, "SHADOW", START)
    for i in range(n_validate):
        entity = f"V{i}"
        published = START + timedelta(days=3 * i)
        _seed_event(conn, entity, published, {"breakout_state": "BREAKOUT_UP"})
        pct = 0.0025 if validate_up else -0.0025
        ohlcv.set_bars(entity, _rising_bars(published, daily_pct=pct))
    test_start = START + timedelta(days=110)
    for i in range(n_test):
        entity = f"T{i}"
        published = test_start + timedelta(days=3 * i)
        _seed_event(conn, entity, published, {"breakout_state": "BREAKOUT_UP"})
        pct = 0.0025 if test_up else -0.0025
        ohlcv.set_bars(entity, _rising_bars(published, daily_pct=pct))


# --- the evidence hierarchy is enforced in order (item 8) ---

def test_rejected_at_statistical_stage_stays_discovered():
    conn = db.connect(":memory:")
    result = _passing_level_result(test_status="REJECTED_NOT_SIGNIFICANT")
    tr = evaluate_relationship_trajectory(conn, FakeOHLCV(), "L2 test", "BREAKOUT", result, 20, BASELINE,
                                           TEST_BOUNDARY, TestIsolationGuard())
    assert tr.reached_state == "DISCOVERED"
    assert "significance test" in tr.rejection_reason


def test_rejected_at_incremental_stage():
    conn = db.connect(":memory:")
    result = _passing_level_result(incremental_status="NO_INCREMENTAL_VALUE")
    tr = evaluate_relationship_trajectory(conn, FakeOHLCV(), "L2 test", "BREAKOUT", result, 20, BASELINE,
                                           TEST_BOUNDARY, TestIsolationGuard())
    assert tr.reached_state == "STATISTICALLY_SUPPORTED"
    assert "Incremental-value" in tr.rejection_reason


def test_rejected_at_shadow_stage_on_failed_permutation():
    conn = db.connect(":memory:")
    result = _passing_level_result(permutation_status="LIKELY_OVERFIT")
    tr = evaluate_relationship_trajectory(conn, FakeOHLCV(), "L2 test", "BREAKOUT", result, 20, BASELINE,
                                           TEST_BOUNDARY, TestIsolationGuard())
    assert tr.reached_state == "INCREMENTAL"
    assert "Permutation" in tr.rejection_reason


def test_rejected_at_shadow_stage_on_unstable_temporal_check():
    conn = db.connect(":memory:")
    result = _passing_level_result(stability_status="UNSTABLE_ACROSS_TIME")
    tr = evaluate_relationship_trajectory(conn, FakeOHLCV(), "L2 test", "BREAKOUT", result, 20, BASELINE,
                                           TEST_BOUNDARY, TestIsolationGuard())
    assert tr.reached_state == "INCREMENTAL"
    assert "Temporal stability" in tr.rejection_reason


def test_reaches_shadow_but_no_relationship_row_exists():
    conn = db.connect(":memory:")  # no validated_relationships row seeded at all
    result = _passing_level_result()
    tr = evaluate_relationship_trajectory(conn, FakeOHLCV(), "L2 test", "BREAKOUT", result, 20, BASELINE,
                                           TEST_BOUNDARY, TestIsolationGuard())
    assert tr.reached_state == "SHADOW"
    assert "No validated_relationships row" in tr.rejection_reason


def test_zero_validate_trades_reports_why_not_just_the_bare_count():
    """Real finding from the stage-7 item-10 real-data run: a relationship
    can be CONFIRMED vs. the crude baseline scalar while its OWN raw
    effect's 95% CI still spans zero - StrategyAgent then abstains on
    EVERY matching row (see strategy_agent.py's CI gate). The rejection
    reason must say so, not just report a bare '0 trades'."""
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    db.upsert_relationship(conn, "rel-ci-spans-zero", CONDITION, 20, -0.01, -0.04, 0.02, 40, "SHADOW", START)
    for i in range(15):
        entity = f"V{i}"
        published = START + timedelta(days=3 * i)
        _seed_event(conn, entity, published, {"breakout_state": "BREAKOUT_UP"})
        ohlcv.set_bars(entity, _rising_bars(published, daily_pct=0.0025))
    result = _passing_level_result()
    tr = evaluate_relationship_trajectory(conn, ohlcv, "L2 test", "BREAKOUT", result, 20, BASELINE, TEST_BOUNDARY,
                                           TestIsolationGuard())
    assert tr.reached_state == "SHADOW"
    assert "0 real VALIDATE-segment trades out of 15 decisions considered" in tr.rejection_reason
    assert "spans zero" in tr.rejection_reason


def test_reaches_economically_supported_but_too_few_test_trades():
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=3)  # below MIN_ECONOMIC_TRADES on TEST side
    result = _passing_level_result()
    tr = evaluate_relationship_trajectory(conn, ohlcv, "L2 test", "BREAKOUT", result, 20, BASELINE, TEST_BOUNDARY,
                                           TestIsolationGuard())
    assert tr.reached_state == "ECONOMICALLY_SUPPORTED"
    assert "TEST-segment trades" in tr.rejection_reason


def test_reaches_economically_supported_but_test_economics_are_negative():
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=15, validate_up=True, test_up=False)
    result = _passing_level_result()
    tr = evaluate_relationship_trajectory(conn, ohlcv, "L2 test", "BREAKOUT", result, 20, BASELINE, TEST_BOUNDARY,
                                           TestIsolationGuard())
    assert tr.reached_state == "ECONOMICALLY_SUPPORTED"
    assert "TEST-segment economics do not confirm" in tr.rejection_reason
    assert tr.test_outcome is not None  # computed and reported even though it didn't confirm


def test_full_pass_reaches_test_validated():
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=15, validate_up=True, test_up=True)
    result = _passing_level_result()
    tr = evaluate_relationship_trajectory(conn, ohlcv, "L2 test", "BREAKOUT", result, 20, BASELINE, TEST_BOUNDARY,
                                           TestIsolationGuard())
    assert tr.reached_state == "TEST_VALIDATED"
    assert tr.rejection_reason is None
    assert tr.relationship_id == "rel-econ"
    assert tr.validate_outcome.n_trades == 15
    assert tr.test_outcome.n_trades == 15
    assert tr.validate_outcome.expectancy > 0
    assert tr.test_outcome.expectancy > 0


# --- the TEST-isolation guard actually gates cross-relationship parameter selection ---

def test_guard_blocks_a_second_relationships_validate_phase_once_test_was_observed():
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=15)
    guard = TestIsolationGuard()

    first = _passing_level_result()
    tr1 = evaluate_relationship_trajectory(conn, ohlcv, "first", "BREAKOUT", first, 20, BASELINE, TEST_BOUNDARY, guard)
    assert tr1.reached_state == "TEST_VALIDATED"
    assert guard.test_observed is True

    # A second, otherwise-independent relationship, evaluated with the SAME guard after the first
    # one's TEST segment was already observed - real per-relationship parameter selection (building
    # its own StrategyAgent/decision process) must be refused.
    db.upsert_relationship(conn, "rel-second", {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                                                  "market_structure": "HIGHER_HIGHS_HIGHER_LOWS"},
                            20, 0.05, 0.01, 0.09, 40, "SHADOW", START)
    second_condition = {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                          "market_structure": "HIGHER_HIGHS_HIGHER_LOWS"}
    second = _passing_level_result(condition=second_condition)
    with pytest.raises(TestIsolationViolation):
        evaluate_relationship_trajectory(conn, ohlcv, "second", "MARKET_STRUCTURE", second, 20, BASELINE,
                                          TEST_BOUNDARY, guard)


# --- the batch orchestrator does NOT lock out later candidates in the same run (unlike two
#     back-to-back single-relationship evaluate_relationship_trajectory calls sharing one guard) ---

def test_batch_orchestrator_lets_multiple_candidates_all_reach_test_validated():
    conn = db.connect(":memory:")
    ohlcv = FakeOHLCV()
    _seed_full_economic_setup(conn, ohlcv, n_validate=15, n_test=15)  # condition 1: breakout_state

    condition2 = {"event_type": "GUIDANCE_CHANGE", "direction": "positive",
                   "market_structure": "HIGHER_HIGHS_HIGHER_LOWS"}
    db.upsert_relationship(conn, "rel-econ-2", condition2, 20, 0.05, 0.01, 0.09, 40, "SHADOW", START)
    for i in range(15):
        entity = f"W{i}"
        published = START + timedelta(days=3 * i)
        event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                             source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                             ingested_at=published, context={"market_structure": "HIGHER_HIGHS_HIGHER_LOWS"})
        pred = PredictionRecord(20, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
        event_id = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, event_id, 0.05, published + timedelta(days=20), 0.03, "OK")
        ohlcv.set_bars(entity, _rising_bars(published, daily_pct=0.0025))
    test_start = START + timedelta(days=110)
    for i in range(15):
        entity = f"U{i}"
        published = test_start + timedelta(days=3 * i)
        event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                             source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                             ingested_at=published, context={"market_structure": "HIGHER_HIGHS_HIGHER_LOWS"})
        pred = PredictionRecord(20, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
        event_id = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, event_id, 0.05, published + timedelta(days=20), 0.03, "OK")
        ohlcv.set_bars(entity, _rising_bars(published, daily_pct=0.0025))

    dummy_l1 = HypothesisTestResult("l1", "CONFIRMED", 40, 0.05, 0.02, 0.001, 0.001)
    research = HierarchicalResearchReport(
        event_type="GUIDANCE_CHANGE", direction="positive", horizon_days=20, budget=DEFAULT_RESEARCH_BUDGET,
        families_screened=2,
        level1_results=[FamilyScreeningResult(concept="BREAKOUT", dimension="breakout_state", test_result=dummy_l1),
                         FamilyScreeningResult(concept="MARKET_STRUCTURE", dimension="market_structure",
                                                test_result=dummy_l1)],
        level2_results={"breakout_state": [_passing_level_result(CONDITION)],
                          "market_structure": [_passing_level_result(condition2)]},
    )

    report = build_stage7_final_report(conn, ohlcv, [research], BASELINE, TEST_BOUNDARY)
    states = {t.label: t.reached_state for t in report.trajectories}
    assert len(states) == 2
    assert all(state == "TEST_VALIDATED" for state in states.values()), states


# --- final report answers (item 9) ---

def test_answers_report_none_survived_when_nothing_clears_any_stage():
    trajectories = [
        RelationshipTrajectory("L2 a", "BREAKOUT", CONDITION, 20, "DISCOVERED", "rejected", None,
                                _passing_level_result(test_status="REJECTED_NOT_SIGNIFICANT")),
    ]
    answers = answer_final_report_questions(trajectories)
    assert "None" in answers["which_relationships_survived_all_controls"]
    assert "None" in answers["which_provide_incremental_information"]
    assert "None" in answers["which_primitives_form_executable_strategies"]
    assert "No" in answers["does_any_advantage_survive_test"]


def test_answers_name_the_surviving_relationship_when_test_validated():
    result = _passing_level_result()
    tr = RelationshipTrajectory("L2 winner", "BREAKOUT", CONDITION, 20, "TEST_VALIDATED", None, "rel-econ", result,
                                 methodology_ids=["meth-1"])
    answers = answer_final_report_questions([tr])
    assert "L2 winner" in answers["which_relationships_survived_all_controls"]
    assert "L2 winner" in answers["which_primitives_form_executable_strategies"]
    assert "meth-1" in answers["which_methodologies_produced_useful_primitives"]
    assert "Yes: L2 winner" in answers["does_any_advantage_survive_test"]


def test_answers_five_way_question_uses_supplied_summary_or_discloses_gap():
    result = _passing_level_result()
    tr = RelationshipTrajectory("L2 a", "BREAKOUT", CONDITION, 20, "DISCOVERED", "x", None, result)
    no_summary = answer_final_report_questions([tr])
    assert "Not assessed" in no_summary["does_adaptive_outperform_static"]

    with_summary = answer_final_report_questions([tr], five_way_summary={"does_adaptive_outperform_static": "Yes, on 3/4 horizons."})
    assert with_summary["does_adaptive_outperform_static"] == "Yes, on 3/4 horizons."


def test_state_counts_cover_every_taxonomy_state():
    result = _passing_level_result()
    tr = RelationshipTrajectory("L2 a", "BREAKOUT", CONDITION, 20, "SHADOW", "x", None, result)
    from market_agent.reporting.stage7_final_report import _state_counts
    counts = _state_counts([tr])
    assert set(counts.keys()) == set(EVIDENCE_STATES)
    assert counts["SHADOW"] == 1


def test_report_to_text_and_to_dict_smoke():
    from market_agent.reporting.stage7_final_report import Stage7FinalReport
    result = _passing_level_result()
    tr = RelationshipTrajectory("L2 a", "BREAKOUT", CONDITION, 20, "SHADOW", "x", None, result)
    report = Stage7FinalReport(generated_at="2024-01-01T00:00:00+00:00", test_boundary=TEST_BOUNDARY,
                                trajectories=[tr], five_way_summary=None,
                                answers=answer_final_report_questions([tr]))
    text = report.to_text()
    assert "SHADOW" in text
    d = report.to_dict()
    assert d["state_counts"]["SHADOW"] == 1
    assert len(d["trajectories"]) == 1
