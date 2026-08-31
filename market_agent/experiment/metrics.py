"""Evaluation metrics for a set of (predicted, realized) pairs - Blueprint
section P / requirement #3: direction accuracy, MAE, RMSE, Spearman
correlation, Brier score/calibration, sample size, bootstrap confidence
intervals.

CONFIDENCE-TO-PROBABILITY IS A DISCLOSED SIMPLIFICATION: neither agent
natively outputs a calibrated probability (Stage 1/2 only produce a point
estimate + a HIGH/MEDIUM/LOW/INSUFFICIENT_PRECEDENT label) - building a
genuinely calibrated probabilistic output is real future work this stage
does not add (see the "do not expand the architecture" instruction this
stage operates under). CONFIDENCE_TO_PROB below is a simple, disclosed,
fixed mapping used ONLY to make a Brier score computable at all; it is
not claimed to be a validated calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DIRECTION_MATTERS_THRESHOLD = 0.005  # same convention as learn/error_taxonomy.py
CONFIDENCE_TO_PROB = {"HIGH": 0.80, "MEDIUM": 0.65, "LOW": 0.55}
N_BOOTSTRAP = 2000
RNG_SEED = 20240601  # fixed seed - bootstrap CIs must be reproducible, not re-random every run


@dataclass
class PredictionOutcome:
    predicted_impact: float | None
    predicted_confidence: str
    realized_abnormal_return: float


@dataclass
class MetricsReport:
    n: int
    direction_accuracy: float | None
    direction_accuracy_ci: tuple[float, float] | None
    mae: float | None
    mae_ci: tuple[float, float] | None
    rmse: float | None
    spearman_r: float | None
    spearman_p: float | None
    brier_score: float | None
    n_excluded_no_prediction: int
    evidence: list[str] = field(default_factory=list)


def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def compute_metrics(outcomes: list[PredictionOutcome]) -> MetricsReport:
    usable = [o for o in outcomes if o.predicted_impact is not None]
    excluded = len(outcomes) - len(usable)
    if len(usable) < 2:
        return MetricsReport(len(usable), None, None, None, None, None, None, None, None, excluded,
                              [f"Only {len(usable)} scoreable prediction(s) - too few for any metric "
                               "to be meaningful."])

    predicted = np.array([o.predicted_impact for o in usable])
    realized = np.array([o.realized_abnormal_return for o in usable])
    errors = predicted - realized

    pred_sign = np.where(predicted > 0, 1, np.where(predicted < 0, -1, 0))
    real_sign = np.where(realized > DIRECTION_MATTERS_THRESHOLD, 1,
                          np.where(realized < -DIRECTION_MATTERS_THRESHOLD, -1, 0))
    directional_mask = (pred_sign != 0) & (real_sign != 0)
    if directional_mask.sum() >= 2:
        correct = (pred_sign[directional_mask] == real_sign[directional_mask]).astype(float)
        direction_accuracy = float(correct.mean())
        direction_ci = _bootstrap_ci(correct, np.mean)
    else:
        direction_accuracy, direction_ci = None, None

    mae = float(np.abs(errors).mean())
    mae_ci = _bootstrap_ci(np.abs(errors), np.mean)
    rmse = float(np.sqrt((errors ** 2).mean()))

    spearman_r, spearman_p = None, None
    if len(usable) >= 3 and np.std(predicted) > 0 and np.std(realized) > 0:
        raw_r, raw_p = stats.spearmanr(predicted, realized)
        # A near-zero-but-not-exactly-zero variance (e.g. STATIC predicting the same signed
        # baseline for every case in a homogeneous subset) can pass the std>0 guard above yet
        # still leave scipy's rank correlation degenerate, returning NaN rather than raising -
        # found running against real data. NaN is not a valid correlation value; report it the
        # same as "not computable" (None -> "n/a"), never print a literal "nan" to the user.
        if not (np.isnan(raw_r) or np.isnan(raw_p)):
            spearman_r, spearman_p = float(raw_r), float(raw_p)

    prob_correct_direction = np.array([CONFIDENCE_TO_PROB.get(o.predicted_confidence, 0.5) for o in usable])
    actual_indicator = (pred_sign == real_sign).astype(float)
    brier = float(np.mean((prob_correct_direction - actual_indicator) ** 2))

    evidence = [f"N={len(usable)} scoreable predictions ({excluded} excluded - no prediction was made).",
                f"Direction accuracy computed on {directional_mask.sum()} cases with a non-trivial "
                f"predicted AND realized sign (threshold {DIRECTION_MATTERS_THRESHOLD:.1%})."]
    return MetricsReport(len(usable), direction_accuracy, direction_ci, mae, mae_ci, rmse, spearman_r,
                          spearman_p, brier, excluded, evidence)
