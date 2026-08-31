"""Tests the four-way harness's OWN new mechanics (four agents scored per
event, DIAGNOSIS_AGENT drives hypothesis generation, STATIC stays frozen,
segments/generalization-case still work with four agents) - the
underlying predict/log/diagnose/hypothesis/test/promote pipeline and its
point-in-time invariants are already proven by test_walkforward.py and
test_integration_loop.py and do not need re-proving here.
"""
from datetime import datetime, timedelta, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.events.schema import RawItem
from market_agent.experiment.four_way_walkforward import (
    AGENT_NAMES, DIAGNOSIS_AGENT, FourWayWalkforwardConfig, run_four_way_walkforward,
)
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.sources.edgar_guidance import SourcedRawItem
from market_agent.store import db

BENCHMARK = "SPY"


class SyntheticPrices(PriceSeriesProvider):
    def __init__(self):
        self.overrides: dict[tuple[str, str], float] = {}

    def set_move(self, entity, event_date, horizon_days, total_return):
        start = event_date.date().isoformat()
        end = (event_date + timedelta(days=horizon_days)).date().isoformat()
        prior = (event_date - timedelta(days=5)).date().isoformat()
        self.overrides[(entity, start)] = 100.0
        self.overrides[(entity, prior)] = 100.0
        self.overrides[(entity, end)] = 100.0 * (1 + total_return)

    def close_price(self, ticker, as_of):
        key = (ticker, as_of.date().isoformat())
        if key in self.overrides:
            return self.overrides[key]
        return 400.0 if ticker == BENCHMARK else 100.0


def _sourced(entity, published, phrase="cuts guidance"):
    return SourcedRawItem(raw_item=RawItem(text=f"{entity} corp: {phrase}", source="SEC EDGAR 8-K", entity=entity,
                                            published_at=published),
                           matched_phrase=phrase, accession_number=f"acc-{entity}-{published.date()}", items_8k=["2.02"])


def test_all_four_agents_are_scored_per_event():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(15):
        d = base + timedelta(days=20 * i)
        items.append(_sourced(f"E{i}", d))
        prices.set_move(f"E{i}", d, 20, -0.03)

    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.2,
                                       final_holdout_fraction=0.2)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    agents_seen = {s.agent for s in report.scored}
    assert agents_seen == set(AGENT_NAMES)
    # every entity that was scored at all should have exactly one prediction per agent
    for entity in {s.entity for s in report.scored}:
        assert {s.agent for s in report.scored if s.entity == entity} == set(AGENT_NAMES)


def test_static_stays_frozen_while_current_adaptive_learns():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    # 40, not 25, PEER events - see test_walkforward.py's identical fixture for why: with
    # store/db.py::deduplicate_by_real_event correctly counting only distinct real events (not once
    # per agent), a hypothesis is tested against the TRUE resolved-so-far count, which peaked at
    # N=14 (one short of MIN_N=15) with only 25 PEER events.
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(40):
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d))
        prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))
    later = base + timedelta(days=100 + 15 * 40 + 60)
    items.append(_sourced("NVDA", later)); prices.set_move("NVDA", later, 20, -0.09)

    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / 46,
                                       final_holdout_fraction=0.0)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    static_predictions = {s.predicted_impact for s in report.scored if s.agent == "STATIC"}
    assert len(static_predictions) == 1  # STATIC never varies, regardless of what CURRENT_ADAPTIVE learned

    nvda_current_adaptive = [s for s in report.scored if s.entity == "NVDA" and s.agent == DIAGNOSIS_AGENT]
    assert len(nvda_current_adaptive) == 1
    assert nvda_current_adaptive[0].generalization_case is True
    assert nvda_current_adaptive[0].basis["basis"] == "validated_relationship"

    nvda_static = [s for s in report.scored if s.entity == "NVDA" and s.agent == "STATIC"][0]
    assert nvda_static.predicted_impact != nvda_current_adaptive[0].predicted_impact


def test_technical_and_methodology_adaptive_fall_back_to_baseline_before_anything_is_learned():
    """With no OHLCV wired into this test (no technical fields ever enter
    context), no concept-conditioned relationship can ever be confirmed -
    TECHNICAL_ADAPTIVE and METHODOLOGY_ADAPTIVE should behave identically
    to the unconditional baseline for the whole run, same as STATIC."""
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = [_sourced("A", base)]
    prices.set_move("A", base, 20, -0.03)

    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.0,
                                       final_holdout_fraction=0.0)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    by_agent = {s.agent: s for s in report.scored if s.entity == "A"}
    assert by_agent["TECHNICAL_ADAPTIVE"].basis["basis"] == "unconditional_baseline"
    assert by_agent["METHODOLOGY_ADAPTIVE"].basis["basis"] == "unconditional_baseline"
    assert by_agent["TECHNICAL_ADAPTIVE"].predicted_impact == by_agent["STATIC"].predicted_impact


def test_segments_are_tagged_for_all_four_agents():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(20):
        d = base + timedelta(days=20 * i)
        items.append(_sourced(f"E{i}", d))
        prices.set_move(f"E{i}", d, 20, -0.03)

    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.2,
                                       final_holdout_fraction=0.2)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    segments = {s.segment for s in report.scored}
    assert segments == {"DEVELOPMENT", "FINAL_HOLDOUT"}
    assert report.n_development > 0 and report.n_final_holdout > 0
    for agent in AGENT_NAMES:
        agent_segments = {s.segment for s in report.scored if s.agent == agent}
        assert agent_segments == {"DEVELOPMENT", "FINAL_HOLDOUT"}


def test_freeze_governance_during_test_blocks_governance_in_the_holdout_segment():
    """Stage 7's TRAIN/VALIDATE/SHADOW/TEST discipline: with
    freeze_governance_during_test=True, no NEW hypothesis, promotion, or
    shadow evaluation may be recorded with a timestamp inside the
    final-holdout segment - predictions still happen there (agents keep
    scoring), but governance stops the moment TEST begins."""
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(40):
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d))
        prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))
    holdout_start = base + timedelta(days=100 + 15 * 40 + 60)
    for i in range(15):
        d = holdout_start + timedelta(days=15 * i)
        items.append(_sourced(f"HOLDOUT{i}", d))
        prices.set_move(f"HOLDOUT{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))

    total = len(items)  # 5 + 40 + 15 = 60
    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / total,
                                       final_holdout_fraction=15 / total, freeze_governance_during_test=True)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    # predictions still happen and get scored during the holdout
    assert report.n_final_holdout > 0

    holdout_predictions = [s for s in report.scored if s.segment == "FINAL_HOLDOUT"]
    test_boundary = min(s.published_at for s in holdout_predictions).isoformat()

    for row in conn.execute("SELECT created_at FROM model_registry").fetchall():
        assert row["created_at"] < test_boundary, \
            f"a governance action ({row['created_at']}) happened at or after the TEST boundary ({test_boundary})"

    for row in conn.execute("SELECT proposed_at FROM candidate_hypotheses").fetchall():
        assert row["proposed_at"] < test_boundary, \
            f"a hypothesis was proposed ({row['proposed_at']}) at or after the TEST boundary ({test_boundary})"


def test_ensemble_adaptive_is_reconfigured_exactly_once_at_the_validate_test_boundary():
    """Harness-level proof for stage 7 item 6 (the isolated AdaptiveAgent/
    make_ensemble_adaptive_agent unit tests in test_agents.py already prove
    the SELECTION mechanism works in isolation - this proves the walk-
    forward harness actually wires it up correctly): ENSEMBLE_ADAPTIVE must
    (a) behave exactly like STATIC for every DEVELOPMENT-segment prediction,
    even after a real relationship has been confirmed and promoted to
    ACTIVE and is already visible to CURRENT_ADAPTIVE; (b) have
    `compute_qualified_relationships_fn` invoked with the live `conn`
    exactly once, at the moment chronological time crosses into
    FINAL_HOLDOUT; and (c) actually USE the relationship the callback
    qualified for every FINAL_HOLDOUT prediction that matches it,
    diverging from STATIC and matching CURRENT_ADAPTIVE instead."""
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(40):
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d))
        prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))
    # 5 MID entities, still inside DEVELOPMENT (well after the relationship should have been
    # promoted), same condition as PEER - proves ENSEMBLE_ADAPTIVE ignores an already-ACTIVE,
    # already-matching relationship until it is explicitly qualified.
    mid_start = base + timedelta(days=100 + 15 * 40 + 60)
    for i in range(5):
        d = mid_start + timedelta(days=10 * i)
        items.append(_sourced(f"MID{i}", d)); prices.set_move(f"MID{i}", d, 20, -0.09)
    # 5 HOLD entities, in FINAL_HOLDOUT, same condition again - proves ENSEMBLE_ADAPTIVE DOES use
    # the relationship once qualified.
    hold_start = mid_start + timedelta(days=10 * 5 + 60)
    for i in range(5):
        d = hold_start + timedelta(days=10 * i)
        items.append(_sourced(f"HOLD{i}", d)); prices.set_move(f"HOLD{i}", d, 20, -0.09)

    conn = db.connect(":memory:")
    # 5 burn-in / 45 development (40 PEER + 5 MID) / 5 final-holdout (HOLD) = 55 total.
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / 55,
                                       final_holdout_fraction=5 / 55)

    call_count = 0

    def compute_qualified(live_conn, test_boundary_iso, unconditional_baseline):
        nonlocal call_count
        call_count += 1
        assert live_conn is conn  # the SAME live connection, not a copy/snapshot
        assert test_boundary_iso == hold_start.isoformat()  # exactly the first HOLD event's published_at
        assert 20 in unconditional_baseline  # the SAME baseline every other agent in this run uses
        rows = live_conn.execute("SELECT relationship_id FROM validated_relationships "
                                  "WHERE status = 'ACTIVE'").fetchall()
        return {r["relationship_id"] for r in rows}

    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn, compute_qualified_relationships_fn=compute_qualified)

    assert call_count == 1

    mid_scores = {s.entity: s for s in report.scored if s.entity.startswith("MID")}
    hold_scores = {s.entity: s for s in report.scored if s.entity.startswith("HOLD")}
    assert len(mid_scores) == 5 and len(hold_scores) == 5

    def by_agent(entity):
        return {s.agent: s for s in report.scored if s.entity == entity}

    # Sanity: the relationship really was confirmed and promoted by MID time, and CURRENT_ADAPTIVE
    # really is using it - otherwise this test would pass vacuously.
    mid_agents = by_agent("MID0")
    assert mid_agents["CURRENT_ADAPTIVE"].basis["basis"] == "validated_relationship"

    # (a) DEVELOPMENT segment: ENSEMBLE_ADAPTIVE == STATIC for every MID entity, despite the
    # relationship already being ACTIVE and visible to CURRENT_ADAPTIVE.
    for entity in mid_scores:
        agents = by_agent(entity)
        assert agents["ENSEMBLE_ADAPTIVE"].predicted_impact == agents["STATIC"].predicted_impact
        assert agents["ENSEMBLE_ADAPTIVE"].predicted_impact != agents["CURRENT_ADAPTIVE"].predicted_impact

    # (c) FINAL_HOLDOUT segment: ENSEMBLE_ADAPTIVE now matches CURRENT_ADAPTIVE (uses the
    # qualified relationship directly) and diverges from STATIC.
    for entity in hold_scores:
        agents = by_agent(entity)
        assert agents["ENSEMBLE_ADAPTIVE"].predicted_impact == agents["CURRENT_ADAPTIVE"].predicted_impact
        assert agents["ENSEMBLE_ADAPTIVE"].predicted_impact != agents["STATIC"].predicted_impact
        assert agents["ENSEMBLE_ADAPTIVE"].basis["basis"] == "validated_relationship"


def test_diagnosis_agent_alone_drives_hypothesis_generation():
    """25 PEER events establish a real RISK_OFF-adjacent pattern strong
    enough to confirm a hypothesis (regime-only conditioning, same proven
    mechanism as stage 1-5) - hypotheses should be generated even though
    only CURRENT_ADAPTIVE's errors are ever diagnosed."""
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(25):
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d))
        prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))

    conn = db.connect(":memory:")
    config = FourWayWalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / 30,
                                       final_holdout_fraction=0.0)
    report = run_four_way_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(),
                                       config, conn)

    assert len(report.promotions) + len(report.rejections) > 0
    hyp_count = conn.execute("SELECT COUNT(*) c FROM candidate_hypotheses").fetchone()["c"]
    assert hyp_count > 0
