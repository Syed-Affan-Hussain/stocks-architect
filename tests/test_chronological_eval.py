from datetime import datetime, timedelta, timezone

from market_agent.experiment.chronological_eval import evaluate_chronologically
from market_agent.experiment.walkforward import ScoredPrediction

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _pred(entity, day, agent, predicted, realized):
    return ScoredPrediction(event_id=f"{entity}-{agent}-{day}", entity=entity, published_at=BASE + timedelta(days=day),
                             horizon_days=20, agent=agent, predicted_impact=predicted, predicted_confidence="MEDIUM",
                             basis={}, realized_abnormal_return=realized, error_type="OK", segment="DEVELOPMENT",
                             generalization_case=False)


def test_no_updates_gives_a_single_window():
    scored = [_pred("A", 0, "STATIC", -0.02, -0.03), _pred("A", 0, "ADAPTIVE", -0.02, -0.03)]
    report = evaluate_chronologically(scored, go_live_timestamps=[])
    assert report.n_updates == 0
    assert len(report.windows) == 1
    assert report.windows[0].n_predictions == 1


def test_one_update_splits_into_two_windows():
    update_time = BASE + timedelta(days=50)
    scored = []
    # before the update: both agents equally mediocre
    for day in range(0, 40, 10):
        scored.append(_pred("A", day, "STATIC", -0.02, -0.06))
        scored.append(_pred("A", day, "ADAPTIVE", -0.02, -0.06))
    # after the update: ADAPTIVE improves, STATIC doesn't (frozen)
    for day in range(60, 100, 10):
        scored.append(_pred("B", day, "STATIC", -0.02, -0.09))
        scored.append(_pred("B", day, "ADAPTIVE", -0.085, -0.09))

    report = evaluate_chronologically(scored, go_live_timestamps=[("rel-1", update_time)])
    assert report.n_updates == 1
    assert len(report.windows) == 2

    before, after = report.windows
    assert before.update_relationship_id is None
    assert after.update_relationship_id == "rel-1"
    assert before.n_predictions == 4
    assert after.n_predictions == 4
    # in the "after" window ADAPTIVE should show a real MAE improvement over STATIC
    assert after.adaptive_mae_improved_vs_static is True
    # in the "before" window both agents were identical - no improvement to show
    assert before.adaptive_mae_improved_vs_static is False


def test_multiple_updates_split_into_multiple_windows():
    t1 = BASE + timedelta(days=30)
    t2 = BASE + timedelta(days=70)
    scored = [_pred("A", d, agent, -0.02, -0.03) for d in (10, 50, 90) for agent in ("STATIC", "ADAPTIVE")]
    report = evaluate_chronologically(scored, go_live_timestamps=[("rel-1", t1), ("rel-2", t2)])
    assert report.n_updates == 2
    assert len(report.windows) == 3
    assert [w.update_relationship_id for w in report.windows] == [None, "rel-1", "rel-2"]
