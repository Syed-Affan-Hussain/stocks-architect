"""Chronological before/after-update evaluation - stage 4 item 9, and the
actual critical test of whether learning helps: not "is ADAPTIVE
different from STATIC in aggregate" but "did ADAPTIVE's relative
performance improve on the predictions that came AFTER each specific
point where new knowledge went live, compared to before it".

A "knowledge update" here means a relationship's `shadow_promoted_at`
timestamp - the moment it started actually influencing live predictions
(AdaptiveAgent only ever queries status='ACTIVE'). Entering SHADOW does
NOT count as an update by this definition, because a SHADOW relationship
has zero effect on any live prediction (learn/shadow.py) - counting it
would measure something that couldn't possibly have changed behavior yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from market_agent.experiment.metrics import MetricsReport, PredictionOutcome, compute_metrics
from market_agent.experiment.walkforward import ScoredPrediction


@dataclass
class UpdateWindow:
    window_index: int
    update_relationship_id: str | None  # None for the FIRST window (before any update ever went live)
    update_timestamp: datetime | None
    window_start: datetime | None
    window_end: datetime | None
    n_predictions: int
    static_metrics: MetricsReport
    adaptive_metrics: MetricsReport
    adaptive_mae_improved_vs_static: bool | None       # None if either metric wasn't computable
    adaptive_direction_improved_vs_static: bool | None


@dataclass
class ChronologicalEvalReport:
    n_updates: int
    windows: list[UpdateWindow] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def evaluate_chronologically(scored: list[ScoredPrediction], go_live_timestamps: list[tuple[str, datetime]]
                              ) -> ChronologicalEvalReport:
    """`go_live_timestamps`: [(relationship_id, shadow_promoted_at), ...] -
    every relationship that ever actually went ACTIVE, in the order it did
    so. Splits `scored` into windows bounded by these timestamps and
    computes STATIC vs. ADAPTIVE metrics independently in each one."""
    go_live_timestamps = sorted(go_live_timestamps, key=lambda t: t[1])
    boundaries = [None] + [t for _, t in go_live_timestamps] + [None]

    windows = []
    for i in range(len(boundaries) - 1):
        window_start, window_end = boundaries[i], boundaries[i + 1]
        window_scored = [s for s in scored
                          if (window_start is None or s.published_at >= window_start)
                          and (window_end is None or s.published_at < window_end)]

        static_outcomes = [PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return)
                            for s in window_scored if s.agent == "STATIC" and s.realized_abnormal_return is not None]
        adaptive_outcomes = [PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return)
                              for s in window_scored if s.agent == "ADAPTIVE" and s.realized_abnormal_return is not None]
        static_metrics = compute_metrics(static_outcomes)
        adaptive_metrics = compute_metrics(adaptive_outcomes)

        mae_improved = (adaptive_metrics.mae < static_metrics.mae
                         if adaptive_metrics.mae is not None and static_metrics.mae is not None else None)
        dir_improved = (adaptive_metrics.direction_accuracy > static_metrics.direction_accuracy
                         if adaptive_metrics.direction_accuracy is not None
                         and static_metrics.direction_accuracy is not None else None)

        rel_id = go_live_timestamps[i - 1][0] if i > 0 else None
        windows.append(UpdateWindow(
            window_index=i, update_relationship_id=rel_id, update_timestamp=window_start,
            window_start=window_start, window_end=window_end,
            n_predictions=len({s.event_id for s in window_scored if s.agent == "ADAPTIVE"}),
            static_metrics=static_metrics, adaptive_metrics=adaptive_metrics,
            adaptive_mae_improved_vs_static=mae_improved, adaptive_direction_improved_vs_static=dir_improved,
        ))

    evidence = [f"{len(go_live_timestamps)} relationship(s) went live (SHADOW -> ACTIVE) during this run, "
                f"splitting the timeline into {len(windows)} evaluation window(s)."]
    if not go_live_timestamps:
        evidence.append("No relationship ever went live in this run - there is only one window (the whole "
                         "run), and it cannot show a before/after effect because nothing changed.")
    return ChronologicalEvalReport(n_updates=len(go_live_timestamps), windows=windows, evidence=evidence)
