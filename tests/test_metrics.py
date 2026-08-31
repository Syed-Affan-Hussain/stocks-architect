from market_agent.experiment.metrics import PredictionOutcome, compute_metrics


def test_perfect_predictions_give_zero_error_and_full_direction_accuracy():
    outcomes = [PredictionOutcome(-0.05, "HIGH", -0.05), PredictionOutcome(0.03, "HIGH", 0.03),
                PredictionOutcome(-0.02, "MEDIUM", -0.02), PredictionOutcome(0.04, "HIGH", 0.04)]
    report = compute_metrics(outcomes)
    assert report.n == 4
    assert report.mae == 0.0
    assert report.rmse == 0.0
    assert report.direction_accuracy == 1.0


def test_opposite_direction_predictions_give_zero_direction_accuracy():
    outcomes = [PredictionOutcome(-0.05, "HIGH", 0.05), PredictionOutcome(0.05, "HIGH", -0.05),
                PredictionOutcome(-0.03, "MEDIUM", 0.03)]
    report = compute_metrics(outcomes)
    assert report.direction_accuracy == 0.0


def test_none_predictions_are_excluded_not_scored_as_zero():
    outcomes = [PredictionOutcome(None, "INSUFFICIENT_PRECEDENT", 0.05), PredictionOutcome(-0.02, "MEDIUM", -0.02)]
    report = compute_metrics(outcomes)
    assert report.n == 1
    assert report.n_excluded_no_prediction == 1


def test_too_few_scoreable_predictions_returns_none_metrics_not_a_crash():
    report = compute_metrics([PredictionOutcome(-0.02, "MEDIUM", -0.03)])
    assert report.n == 1
    assert report.mae is None
    assert report.direction_accuracy is None


def test_confidence_interval_brackets_the_point_estimate():
    outcomes = [PredictionOutcome(-0.05, "HIGH", -0.04), PredictionOutcome(-0.04, "HIGH", -0.06),
                PredictionOutcome(-0.06, "HIGH", -0.03), PredictionOutcome(-0.05, "HIGH", -0.05),
                PredictionOutcome(-0.03, "HIGH", -0.07), PredictionOutcome(-0.07, "HIGH", -0.02),
                PredictionOutcome(-0.05, "HIGH", -0.04), PredictionOutcome(-0.04, "HIGH", -0.06),
                PredictionOutcome(-0.06, "HIGH", -0.03), PredictionOutcome(-0.05, "HIGH", -0.05)]
    report = compute_metrics(outcomes)
    lo, hi = report.mae_ci
    assert lo <= report.mae <= hi


def test_brier_score_is_between_zero_and_one():
    outcomes = [PredictionOutcome(-0.05, "HIGH", -0.04), PredictionOutcome(0.03, "LOW", -0.01),
                PredictionOutcome(-0.02, "MEDIUM", 0.02)]
    report = compute_metrics(outcomes)
    assert 0.0 <= report.brier_score <= 1.0


def test_spearman_reported_as_none_not_nan_when_scipy_returns_nan(monkeypatch):
    """A frozen agent predicting nearly the same value for every case can leave scipy's rank
    correlation returning NaN even though the std>0 guard passed - found running against real
    data. Must surface as None ("n/a"), never a literal NaN string. Tested by forcing scipy's
    return value directly, rather than trying to reverse-engineer scipy's exact internal
    degeneracy conditions synthetically."""
    import market_agent.experiment.metrics as metrics_module

    monkeypatch.setattr(metrics_module.stats, "spearmanr", lambda a, b: (float("nan"), float("nan")))
    outcomes = [PredictionOutcome(-0.05, "HIGH", -0.04), PredictionOutcome(0.03, "LOW", -0.01),
                PredictionOutcome(-0.02, "MEDIUM", 0.02)]
    report = compute_metrics(outcomes)
    assert report.spearman_r is None
    assert report.spearman_p is None
