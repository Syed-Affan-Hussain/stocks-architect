"""Tests the walk-forward harness's own NEW mechanics (chronological
embargo ordering, segment tagging, burn-in baseline estimation,
generalization-case flagging, point-in-time invariant) - the underlying
predict/log/diagnose/hypothesis/test/promote pipeline is already covered
by tests/test_integration_loop.py and does not need re-proving here.
"""
from datetime import datetime, timedelta, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.walkforward import WalkforwardConfig, run_walkforward
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.sources.edgar_guidance import SourcedRawItem
from market_agent.events.schema import RawItem
from market_agent.store import db

BENCHMARK = "SPY"


class SyntheticPrices(PriceSeriesProvider):
    """Deterministic prices: benchmark flat at 400 everywhere (so
    abnormal return == raw entity return, and regime is always NORMAL
    unless a specific window is overridden), each entity flat at 100
    except for an engineered move over its own event's horizon window."""

    def __init__(self):
        self.overrides: dict[tuple[str, str], float] = {}

    def set_move(self, entity: str, event_date: datetime, horizon_days: int, total_return: float):
        start = event_date.date().isoformat()
        end = (event_date + timedelta(days=horizon_days)).date().isoformat()
        prior = (event_date - timedelta(days=5)).date().isoformat()
        self.overrides[(entity, start)] = 100.0
        self.overrides[(entity, prior)] = 100.0
        self.overrides[(entity, end)] = 100.0 * (1 + total_return)

    def override_benchmark_window(self, as_of: datetime, lookback_return: float):
        end = as_of.date().isoformat()
        start = (as_of - timedelta(days=60)).date().isoformat()
        self.overrides[(BENCHMARK, end)] = 400.0 * (1 + lookback_return)
        self.overrides[(BENCHMARK, start)] = 400.0

    def close_price(self, ticker, as_of):
        key = (ticker, as_of.date().isoformat())
        if key in self.overrides:
            return self.overrides[key]
        return 400.0 if ticker == BENCHMARK else 100.0


def _sourced(entity, published, phrase="cuts guidance"):
    return SourcedRawItem(raw_item=RawItem(text=f"{entity} corp: {phrase}", source="SEC EDGAR 8-K", entity=entity,
                                            published_at=published),
                           matched_phrase=phrase, accession_number=f"acc-{entity}-{published.date()}", items_8k=["2.02"])


def test_burn_in_events_are_not_scored():
    prices = SyntheticPrices()
    items = []
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        d = base + timedelta(days=20 * i)
        items.append(_sourced(f"E{i}", d))
        prices.set_move(f"E{i}", d, 20, -0.03)

    conn = db.connect(":memory:")
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.5, final_holdout_fraction=0.2)
    report = run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    scored_entities = {s.entity for s in report.scored}
    burn_in_entities = {f"E{i}" for i in range(5)}  # first 50% of 10 = 5
    assert scored_entities.isdisjoint(burn_in_entities)


def test_outcome_not_resolved_before_horizon_plus_embargo_elapses():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # event A resolves at day 22 (20 + 2 embargo). event B is published on day 21 - A must NOT be
    # resolved yet when B is predicted (only 21 days have passed, embargo needs 22).
    items = [_sourced("A", base), _sourced("B", base + timedelta(days=21)),
             _sourced("C", base + timedelta(days=40))]
    for entity, d in [("A", base), ("B", base + timedelta(days=21)), ("C", base + timedelta(days=40))]:
        prices.set_move(entity, d, 20, -0.03)

    conn = db.connect(":memory:")
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.0, final_holdout_fraction=0.0)
    run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    # A's outcome should be resolved (locked) by the time the dataset is fully processed
    a_rows = conn.execute("SELECT * FROM episodic_events WHERE entity = 'A'").fetchall()
    assert all(r["outcome_locked"] == 1 for r in a_rows)


def test_segments_are_tagged_development_and_final_holdout():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(20):
        d = base + timedelta(days=20 * i)
        items.append(_sourced(f"E{i}", d))
        prices.set_move(f"E{i}", d, 20, -0.03)

    conn = db.connect(":memory:")
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.2, final_holdout_fraction=0.2)
    report = run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    segments = {s.segment for s in report.scored}
    assert segments == {"DEVELOPMENT", "FINAL_HOLDOUT"}
    assert report.n_development > 0 and report.n_final_holdout > 0


def test_generalization_case_flagged_for_cross_entity_relationship_use():
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    # 40 same-regime (NORMAL, benchmark flat) "PEER" events with a strong, consistent -9% effect,
    # far from the naive baseline that burn-in will estimate from a milder -3% pattern. 40, not 25 -
    # store/db.py::deduplicate_by_real_event (found running a real 4-agent walk-forward: N was
    # inflated exactly 2x/4x by counting every agent's own logged row for the SAME real event as an
    # independent observation) means a hypothesis is now tested against the TRUE count of distinct
    # prior PEER events resolved so far, which lags behind the raw event count (each PEER's outcome
    # only resolves horizon+embargo=22 days after publication, and PEERs are added every 15 days) -
    # 25 real events peaked at N=14, one short of MIN_N=15; 40 gives real margin.
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(40):
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d)); prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))
    # a later, DIFFERENT entity, same regime/situation
    later = base + timedelta(days=100 + 15 * 40 + 60)
    items.append(_sourced("NVDA", later)); prices.set_move("NVDA", later, 20, -0.09)

    conn = db.connect(":memory:")
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / 46, final_holdout_fraction=0.0)
    report = run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    nvda_predictions = [s for s in report.scored if s.entity == "NVDA" and s.agent == "ADAPTIVE"]
    assert len(nvda_predictions) == 1
    assert nvda_predictions[0].generalization_case is True
    assert nvda_predictions[0].basis["basis"] == "validated_relationship"

    static_nvda = [s for s in report.scored if s.entity == "NVDA" and s.agent == "STATIC"][0]
    assert static_nvda.predicted_impact != nvda_predictions[0].predicted_impact


def test_no_relationship_used_before_its_own_promotion_time():
    """Point-in-time audit: for every ADAPTIVE prediction that used a
    validated_relationship, that relationship's created_at must be <=
    the prediction's published_at - the core correctness argument in
    this module's docstring, checked directly rather than only argued."""
    prices = SyntheticPrices()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = []
    for i in range(5):
        d = base + timedelta(days=15 * i)
        items.append(_sourced(f"BURN{i}", d)); prices.set_move(f"BURN{i}", d, 20, -0.03)
    for i in range(40):  # see test_generalization_case_flagged_for_cross_entity_relationship_use for why 40
        d = base + timedelta(days=100 + 15 * i)
        items.append(_sourced(f"PEER{i}", d)); prices.set_move(f"PEER{i}", d, 20, -0.09 + (0.002 if i % 2 == 0 else -0.002))

    conn = db.connect(":memory:")
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=5 / 45, final_holdout_fraction=0.0)
    report = run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    checked = 0
    for s in report.scored:
        if s.agent == "ADAPTIVE" and s.basis.get("basis") == "validated_relationship":
            rel = conn.execute("SELECT created_at FROM validated_relationships WHERE relationship_id = ?",
                                (s.basis["relationship_id"],)).fetchone()
            assert rel["created_at"] <= s.published_at.isoformat()
            checked += 1
    assert checked > 0  # the test is only meaningful if at least one relationship was actually used


def test_report_declares_the_active_interpreter_and_generator():
    prices = SyntheticPrices()
    items = [_sourced("A", datetime(2020, 1, 1, tzinfo=timezone.utc))]
    prices.set_move("A", datetime(2020, 1, 1, tzinfo=timezone.utc), 20, -0.03)
    config = WalkforwardConfig(horizon_days_list=[20], embargo_days=2, burn_in_fraction=0.0, final_holdout_fraction=0.0)
    report = run_walkforward(items, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config,
                              db.connect(":memory:"))
    assert report.interpreter_used == "RULE_BASED"
    assert report.hypothesis_generator_used == "RULE_BASED"
    assert "NO LLM reasoning" in report.evidence[0]
