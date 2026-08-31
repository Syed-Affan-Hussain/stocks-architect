from datetime import datetime, timedelta, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.learn.overfitting_diagnostics import (
    run_permutation_test as run_perm_test, run_temporal_stability_check as run_stability_check,
)
from market_agent.store import db

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _log(conn, entity, published, realized):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                         source_reliability_snapshot=0.5, raw_text="raises guidance", published_at=published,
                         ingested_at=published, context={"regime": "RISK_ON"})
    pred = PredictionRecord(20, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, realized, published + timedelta(days=20), realized - 0.02, "OK")
    return db.get_event(conn, event_id)


# --- permutation test ---

def test_permutation_insufficient_n():
    conn = db.connect(":memory:")
    matching = [_log(conn, f"E{i}", NOW + timedelta(days=i), 0.05) for i in range(5)]
    result = run_perm_test(matching, matching)
    assert result.status == "INSUFFICIENT_N"


def test_permutation_survives_when_subset_is_a_real_outlier_vs_the_pool():
    conn = db.connect(":memory:")
    # pool: 100 events with returns clustered tightly around 0.0
    pool = [_log(conn, f"POOL{i}", NOW + timedelta(days=i), 0.001 if i % 2 == 0 else -0.001) for i in range(100)]
    # matching subset: 20 of them replaced with a strongly, consistently different return
    matching = [_log(conn, f"SUBSET{i}", NOW + timedelta(days=200 + i), 0.15 + (0.005 if i % 2 == 0 else -0.005))
                for i in range(20)]
    full_pool = pool + matching
    result = run_perm_test(matching, full_pool, n_permutations=500)
    assert result.status == "SURVIVES_PERMUTATION"
    assert result.permutation_p_value < 0.05


def test_permutation_likely_overfit_when_subset_looks_like_a_random_draw():
    conn = db.connect(":memory:")
    # pool and "matching" subset are drawn from the SAME distribution - no real selection effect
    import random
    rng = random.Random(42)
    pool = [_log(conn, f"POOL{i}", NOW + timedelta(days=i), rng.uniform(-0.02, 0.02)) for i in range(200)]
    matching = pool[:15]  # a genuinely random-looking subset of the same pool
    result = run_perm_test(matching, pool, n_permutations=500)
    assert result.status == "LIKELY_OVERFIT"


def test_permutation_pool_smaller_than_matching_is_insufficient():
    conn = db.connect(":memory:")
    matching = [_log(conn, f"E{i}", NOW + timedelta(days=i), 0.05) for i in range(20)]
    tiny_pool = matching[:5]
    result = run_perm_test(matching, tiny_pool)
    assert result.status == "INSUFFICIENT_N"


def test_permutation_is_deterministic_given_the_fixed_seed():
    conn = db.connect(":memory:")
    pool = [_log(conn, f"POOL{i}", NOW + timedelta(days=i), 0.001 * (i % 5)) for i in range(60)]
    matching = pool[:20]
    r1 = run_perm_test(matching, pool, n_permutations=300)
    r2 = run_perm_test(matching, pool, n_permutations=300)
    assert r1.permutation_p_value == r2.permutation_p_value


# --- temporal stability ---

def test_stability_insufficient_n():
    conn = db.connect(":memory:")
    matching = [_log(conn, f"E{i}", NOW + timedelta(days=i), 0.05) for i in range(5)]
    result = run_stability_check(matching)
    assert result.status == "INSUFFICIENT_N"


def test_stability_stable_when_sign_is_consistent_across_time():
    conn = db.connect(":memory:")
    matching = [_log(conn, f"E{i}", NOW + timedelta(days=10 * i), 0.05 + (0.002 if i % 2 == 0 else -0.002))
                for i in range(30)]
    result = run_stability_check(matching)
    assert result.status == "STABLE_ACROSS_TIME"
    assert result.same_sign is True


def test_stability_unstable_when_sign_flips_across_time():
    conn = db.connect(":memory:")
    first_half = [_log(conn, f"EARLY{i}", NOW + timedelta(days=10 * i), 0.08) for i in range(15)]
    second_half = [_log(conn, f"LATE{i}", NOW + timedelta(days=500 + 10 * i), -0.08) for i in range(15)]
    result = run_stability_check(first_half + second_half)
    assert result.status == "UNSTABLE_ACROSS_TIME"
    assert result.same_sign is False


def test_stability_never_gates_promotion_only_reports():
    conn = db.connect(":memory:")
    matching = [_log(conn, f"E{i}", NOW + timedelta(days=10 * i), 0.05) for i in range(20)]
    result = run_stability_check(matching)
    assert "STABLE" in result.status or "UNSTABLE" in result.status or "INSUFFICIENT" in result.status
