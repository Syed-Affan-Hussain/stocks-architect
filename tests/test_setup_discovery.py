from datetime import datetime, timedelta, timezone

from market_agent.concepts.technical_context import TECHNICAL_STATE_FIELD_NAMES
from market_agent.setups.setup_discovery import (
    DEFAULT_SETUP_SEARCH_BUDGET, SetupSearchBudget, run_setup_discovery_pass, run_setup_level1_screening,
    segment_observations,
)
from market_agent.store import db

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
HORIZON = 20


def _technical(breakout_state="UNKNOWN", **overrides):
    d = {name: "UNKNOWN" for name in TECHNICAL_STATE_FIELD_NAMES}
    d["breakout_state"] = breakout_state
    d.update(overrides)
    return d


def _seed(conn, entity, as_of, breakout_state, realized, regime="NORMAL", horizon_days=HORIZON, **tech_overrides):
    obs_id = db.log_setup_observation(conn, entity, as_of, regime, _technical(breakout_state, **tech_overrides),
                                       horizon_days)
    db.record_setup_outcome(conn, obs_id, realized, as_of + timedelta(days=horizon_days))


def _seed_mixed_population(conn, n=300, matching_effect=0.08, default_effect=0.0, horizon_days=HORIZON):
    """Half BREAKOUT_UP with a strong positive effect, half NONE (default)
    near zero - spread evenly across the whole chronological range so
    every TRAIN/VALIDATE/SHADOW/TEST segment gets a proportional mix of
    both, and the matching subset's mean differs meaningfully from the
    TRAIN-derived pooled baseline (which sits between the two)."""
    for i in range(n):
        d = START + timedelta(days=i)
        if i % 2 == 0:
            jitter = 0.002 if i % 4 == 0 else -0.002
            _seed(conn, f"E{i}", d, "BREAKOUT_UP", matching_effect + jitter, horizon_days=horizon_days)
        else:
            jitter = 0.001 if i % 4 == 1 else -0.001
            _seed(conn, f"E{i}", d, "NONE", default_effect + jitter, horizon_days=horizon_days)


def test_segment_observations_splits_chronologically_and_computes_train_baseline():
    conn = db.connect(":memory:")
    _seed_mixed_population(conn, n=200)
    rows = db.query_setup_observations(conn, horizon_days=HORIZON, outcome_known_only=True)
    segmented = segment_observations(rows)

    assert len(segmented.train) == 80    # 40% of 200
    assert len(segmented.validate) == 60  # 30%
    assert len(segmented.shadow) == 30    # 15%
    assert len(segmented.test) == 30      # remainder (15%)
    assert segmented.train_baseline_mean is not None
    assert 0.0 < segmented.train_baseline_mean < 0.08  # between the two population means


def test_segment_observations_empty_input():
    segmented = segment_observations([])
    assert segmented.train == [] and segmented.train_baseline_mean is None


def test_level1_screening_confirms_a_real_signal_dimension():
    conn = db.connect(":memory:")
    _seed_mixed_population(conn, n=300)
    rows = db.query_setup_observations(conn, horizon_days=HORIZON, outcome_known_only=True)
    segmented = segment_observations(rows)

    results, dropped = run_setup_level1_screening(segmented.train, segmented.train_baseline_mean,
                                                    DEFAULT_SETUP_SEARCH_BUDGET)
    by_dim = dict(results)
    assert by_dim["breakout_state"].status == "CONFIRMED"
    assert dropped == []


def test_level1_screening_rejects_insufficient_n():
    conn = db.connect(":memory:")
    _seed_mixed_population(conn, n=20)  # too few per-segment/per-dimension observations
    rows = db.query_setup_observations(conn, horizon_days=HORIZON, outcome_known_only=True)
    segmented = segment_observations(rows)
    results, _ = run_setup_level1_screening(segmented.train, segmented.train_baseline_mean, DEFAULT_SETUP_SEARCH_BUDGET)
    by_dim = dict(results)
    assert by_dim["breakout_state"].status == "INSUFFICIENT_N"


def test_budget_drops_dimensions_beyond_the_cap():
    conn = db.connect(":memory:")
    _seed_mixed_population(conn, n=300)
    rows = db.query_setup_observations(conn, horizon_days=HORIZON, outcome_known_only=True)
    segmented = segment_observations(rows)
    small_budget = SetupSearchBudget(max_single_dimensions_screened=2, max_dimensions_per_combination=2,
                                      max_combinations_tested=10, label="test_small")
    results, dropped = run_setup_level1_screening(segmented.train, segmented.train_baseline_mean, small_budget)
    assert len(results) == 2
    assert len(dropped) == 17  # 19 screenable dimensions - 2 tested


def test_full_pass_reaches_test_validated_on_a_real_persistent_signal():
    conn = db.connect(":memory:")
    _seed_mixed_population(conn, n=400, matching_effect=0.08, default_effect=0.0)
    report = run_setup_discovery_pass(conn, HORIZON, created_at=START)

    assert report.n_observations == 400
    winners = [s for s in report.setups if s.status == "TEST_VALIDATED"]
    assert winners, [ (s.status, s.technical_conditions, s.train_result.status if s.train_result else None)
                      for s in report.setups ]
    winner = winners[0]
    assert winner.technical_conditions.get("breakout_state") == "BREAKOUT_UP"
    assert winner.train_result.status == "CONFIRMED"
    assert winner.validate_result.status == "CONFIRMED"
    assert winner.shadow_result.status == "CONFIRMED"
    assert winner.test_result.status == "CONFIRMED"

    # persisted to the store, not just in-memory
    row = db.get_discovered_setup(conn, winner.setup_id)
    assert row is not None
    assert row["status"] == "TEST_VALIDATED"


def test_signal_only_in_train_is_rejected_at_validate_and_stops_early():
    conn = db.connect(":memory:")
    # Strong signal for the first 40% (TRAIN) only - VALIDATE/SHADOW/TEST portions have NO signal at
    # all (both branches share the same near-zero effect), so this must be rejected once it reaches
    # VALIDATE testing, and never even look at SHADOW/TEST.
    n = 400
    n_train = int(n * 0.40)
    for i in range(n):
        d = START + timedelta(days=i)
        if i < n_train and i % 2 == 0:
            _seed(conn, f"E{i}", d, "BREAKOUT_UP", 0.08 + (0.002 if i % 4 == 0 else -0.002))
        elif i < n_train:
            _seed(conn, f"E{i}", d, "NONE", 0.0 + (0.001 if i % 4 == 1 else -0.001))
        else:
            # no signal at all post-TRAIN: both branches get the SAME near-zero effect
            state = "BREAKOUT_UP" if i % 2 == 0 else "NONE"
            _seed(conn, f"E{i}", d, state, 0.0 + (0.001 if i % 3 == 0 else -0.001))

    report = run_setup_discovery_pass(conn, HORIZON, created_at=START)
    breakout_setups = [s for s in report.setups if s.technical_conditions.get("breakout_state") == "BREAKOUT_UP"]
    assert breakout_setups
    setup = breakout_setups[0]
    assert setup.status == "REJECTED"
    assert setup.train_result.status == "CONFIRMED"
    assert setup.validate_result is not None
    # VALIDATE's own raw t-test can still come out statistically "CONFIRMED" (its near-zero mean
    # genuinely differs from the TRAIN-derived baseline) - that number is reported honestly, unedited.
    # What must fail is ESCALATION: VALIDATE's deviation from baseline is in the OPPOSITE direction
    # from TRAIN's, which is a sign flip, not a re-confirmation (see _escalates in setup_discovery.py).
    train_direction = setup.train_result.mean_effect - setup.train_result.baseline
    validate_direction = setup.validate_result.mean_effect - setup.validate_result.baseline
    assert train_direction * validate_direction < 0
    assert setup.shadow_result is None  # never reached - stopped at VALIDATE
    assert setup.test_result is None


def test_regime_is_split_out_of_technical_conditions_when_part_of_a_combo():
    """A Level-2 combination CAN include 'regime' alongside a technical
    dimension - the resulting Setup must carry it in `.regime`, never
    inside `.technical_conditions` (see module docstring's REGIME/SETUP
    split)."""
    conn = db.connect(":memory:")
    n = 400
    for i in range(n):
        d = START + timedelta(days=i)
        # regime correlates perfectly with breakout_state here, so 'regime' ALSO screens in at
        # Level 1 on its own (its RISK_ON subset is exactly the informative half of the population).
        if i % 2 == 0:
            _seed(conn, f"E{i}", d, "BREAKOUT_UP", 0.08 + (0.002 if i % 4 == 0 else -0.002), regime="RISK_ON")
        else:
            _seed(conn, f"E{i}", d, "NONE", 0.0 + (0.001 if i % 4 == 1 else -0.001), regime="NORMAL")

    budget = SetupSearchBudget(max_single_dimensions_screened=19, max_dimensions_per_combination=2,
                                max_combinations_tested=40, label="test")
    report = run_setup_discovery_pass(conn, HORIZON, created_at=START, budget=budget)
    # regime=RISK_ON is constant here so it screens in at Level 1 (never the default NORMAL/UNKNOWN)
    # and should combine with breakout_state.
    combo_setups = [s for s in report.setups if s.technical_conditions.get("breakout_state") == "BREAKOUT_UP"
                     and s.regime == "RISK_ON"]
    assert combo_setups
    for s in combo_setups:
        assert "regime" not in s.technical_conditions
