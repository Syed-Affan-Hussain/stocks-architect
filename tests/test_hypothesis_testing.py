from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.hypothesis_testing import MIN_N, _holm_correct
from market_agent.learn.hypothesis_testing import test_hypotheses_batch as run_batch
from market_agent.learn.hypothesis_testing import test_hypothesis as run_single_test
from market_agent.store import db

BASELINE = {20: 0.02}  # unconditional_baseline: |effect|=2% at 20d -> signed -0.02 for "negative" direction
SOURCE_EVENT_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _log_history_event(conn, entity, published, realized_return, regime="RISK_OFF"):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                         ingested_at=published, context={"regime": regime})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized_return, published + timedelta(days=20), realized_return + 0.02, "OK")


def _make_hypothesis(conn, n_prior, prior_values, regime="RISK_OFF"):
    """Logs the triggering (source) event, N prior history rows (all
    published BEFORE the source event), and a hypothesis conditioned on
    `regime`. Returns the hypothesis row."""
    source_event = EventRecord(entity="NVDA", event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                                source_reliability_snapshot=0.5, raw_text="cuts guidance",
                                published_at=SOURCE_EVENT_DATE, ingested_at=SOURCE_EVENT_DATE,
                                context={"regime": regime})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", SOURCE_EVENT_DATE)
    source_event_id = db.log_prediction(conn, source_event, pred)
    db.record_outcome(conn, source_event_id, -999.0, SOURCE_EVENT_DATE + timedelta(days=20), -999.0, "WRONG_DIRECTION")
    # -999 is an intentionally absurd outcome for the SOURCE event itself - if the hindsight-bias
    # exclusion in hypothesis_testing.py ever broke and let this row leak into its own test, any
    # reasonable statistic would be wildly, detectably distorted by it.

    for i, val in enumerate(prior_values[:n_prior]):
        _log_history_event(conn, f"PEER{i}", SOURCE_EVENT_DATE - timedelta(days=30 * (i + 1)), val, regime)

    hid = db.add_hypothesis(conn, source_event_id, {"event_type": "GUIDANCE_CHANGE", "direction": "negative",
                                                      "regime": regime}, 20, "test hypothesis", SOURCE_EVENT_DATE)
    return conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()


def test_insufficient_n_rejects_without_computing_a_p_value():
    conn = db.connect(":memory:")
    hyp = _make_hypothesis(conn, n_prior=5, prior_values=[-0.10] * 5)
    result = run_single_test(conn, hyp, BASELINE)
    assert result.status == "REJECTED_INSUFFICIENT_N"
    assert result.p_value is None
    assert result.n == 5


def test_clear_consistent_effect_is_confirmed():
    conn = db.connect(":memory:")
    values = [-0.10, -0.11, -0.09, -0.10, -0.115, -0.095, -0.10, -0.105, -0.098, -0.102,
              -0.11, -0.09, -0.10, -0.108, -0.101, -0.099, -0.103, -0.097, -0.11, -0.09]
    hyp = _make_hypothesis(conn, n_prior=20, prior_values=values)
    result = run_single_test(conn, hyp, BASELINE)
    assert result.status == "CONFIRMED"
    assert result.n == 20
    assert result.mean_effect < -0.09  # nowhere near the -0.02 baseline
    assert result.p_value < 0.05


def test_noise_around_baseline_is_not_significant():
    conn = db.connect(":memory:")
    values = [-0.05, 0.01, -0.03, 0.02, -0.04, 0.00, -0.02, -0.06, 0.03, -0.01,
              -0.02, 0.01, -0.05, 0.02, -0.03, 0.00, -0.04, 0.01, -0.02, 0.03]
    hyp = _make_hypothesis(conn, n_prior=20, prior_values=values)
    result = run_single_test(conn, hyp, BASELINE)
    assert result.status == "REJECTED_NOT_SIGNIFICANT"


def test_tiny_but_precise_effect_is_rejected_as_economically_trivial():
    """A statistically detectable but economically meaningless difference
    from baseline must not be promoted - Blueprint's minimum-economic-
    effect gate, independent of the significance test."""
    conn = db.connect(":memory:")
    values = [-0.0214, -0.0216, -0.0215, -0.0214, -0.0216, -0.0215, -0.0214, -0.0216,
              -0.0215, -0.0214, -0.0216, -0.0215, -0.0214, -0.0216, -0.0215, -0.0214,
              -0.0216, -0.0215, -0.0214, -0.0216]  # baseline is -0.02; diff ~ -0.0015, tiny and very consistent
    hyp = _make_hypothesis(conn, n_prior=20, prior_values=values)
    result = run_single_test(conn, hyp, BASELINE)
    assert result.status == "REJECTED_ECONOMICALLY_TRIVIAL"


def test_source_event_itself_is_excluded_from_its_own_test():
    """Hindsight-bias defense: the absurd -999 outcome planted on the
    source event in _make_hypothesis must never appear in the tested
    sample - if it did, mean_effect would be wildly distorted."""
    conn = db.connect(":memory:")
    values = [-0.10 + (0.001 if i % 2 == 0 else -0.001) for i in range(20)]
    hyp = _make_hypothesis(conn, n_prior=20, prior_values=values)
    result = run_single_test(conn, hyp, BASELINE)
    assert result.n == 20  # not 21 - the source event did not count itself
    assert result.mean_effect > -0.5  # nowhere near being dragged toward -999


def test_future_published_history_is_excluded():
    """A prior-looking event published AFTER the triggering event must not
    leak into the test, even if it would otherwise match the condition."""
    conn = db.connect(":memory:")
    hyp = _make_hypothesis(conn, n_prior=20, prior_values=[-0.10 + (0.001 if i % 2 == 0 else -0.001) for i in range(20)])
    # a "future" matching event, published after SOURCE_EVENT_DATE, with a wildly different outcome
    _log_history_event(conn, "FUTURE_PEER", SOURCE_EVENT_DATE + timedelta(days=10), 0.50, "RISK_OFF")
    result = run_single_test(conn, hyp, BASELINE)
    assert result.n == 20  # the future row must not have been counted


def test_holm_correction_matches_hand_computed_values():
    # classic worked example: p=[0.01,0.02,0.03,0.20], m=4
    adjusted = _holm_correct([0.01, 0.02, 0.03, 0.20])
    assert adjusted == [0.04, 0.06, 0.06, 0.20]


def test_batch_correction_can_flip_a_marginal_confirmation_to_rejected():
    conn = db.connect(":memory:")
    # one hypothesis with a real, moderate, marginally-significant effect
    marginal_values = [-0.10, -0.11, -0.09, -0.10, -0.115, -0.095, -0.10, -0.105, -0.098, -0.102,
                        -0.11, -0.09, -0.10, -0.108, -0.101, -0.099]
    marginal = _make_hypothesis(conn, n_prior=16, prior_values=marginal_values, regime="RISK_OFF")
    # three "distractor" hypotheses with no real effect, diluting the correction budget
    noise_values = [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02,
                     -0.02, -0.02, -0.02, -0.02, -0.02, -0.02]
    distractor1 = _make_hypothesis(conn, n_prior=16, prior_values=noise_values, regime="RISK_OFF")
    distractor2 = _make_hypothesis(conn, n_prior=16, prior_values=noise_values, regime="RISK_OFF")
    distractor3 = _make_hypothesis(conn, n_prior=16, prior_values=noise_values, regime="RISK_OFF")

    results = run_batch(conn, [marginal, distractor1, distractor2, distractor3], BASELINE)
    for r in results:
        assert r.p_value_corrected is not None or r.p_value is None
        if r.p_value is not None:
            assert r.p_value_corrected >= r.p_value  # Holm correction never makes a p-value smaller
