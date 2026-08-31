"""Runs the walk-forward Static-vs-Adaptive experiment against REAL SEC
EDGAR guidance-change events and REAL Yahoo Finance prices - stage 4:
multiple horizons, shadow deployment, expanded context, and chronological
before/after-update evaluation.

    python scripts/run_real_experiment.py

This configuration (horizons, embargo, burn-in/holdout fractions, and
every statistical constant in market_agent/learn/hypothesis_testing.py,
market_agent/learn/shadow.py, and market_agent/retrieval/regime.py) was
fixed BEFORE this script was run against real data under this
configuration and is not to be retuned based on what this run reports -
see walkforward.py's module docstring. Whatever this prints is reported
as-is, including a negative result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from datetime import datetime, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.chronological_eval import evaluate_chronologically
from market_agent.experiment.metrics import PredictionOutcome, compute_metrics
from market_agent.experiment.walkforward import WalkforwardConfig, run_walkforward
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.pit.clock import PointInTimeClock
from market_agent.reporting.knowledge_state import build_knowledge_state_report
from market_agent.sources.edgar_guidance import fetch_guidance_events
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
CACHE_PATH = "data_cache/edgar/guidance_2016_2024.json"
HORIZONS = [1, 5, 20, 60]


def _print_metrics_block(title: str, outcomes: list[PredictionOutcome]) -> None:
    m = compute_metrics(outcomes)
    print(f"\n  {title} (n={m.n}, {m.n_excluded_no_prediction} excluded - no prediction made)")
    if m.n < 2:
        print("    Too few scoreable predictions for metrics.")
        return
    if m.direction_accuracy is not None:
        ci = f"  95% CI [{m.direction_accuracy_ci[0]:.1%}, {m.direction_accuracy_ci[1]:.1%}]" if m.direction_accuracy_ci else ""
        print(f"    Direction accuracy: {m.direction_accuracy:.1%}{ci}")
    else:
        print("    Direction accuracy: n/a (not enough non-trivial-sign cases)")
    print(f"    MAE: {m.mae:.2%}  95% CI [{m.mae_ci[0]:.2%}, {m.mae_ci[1]:.2%}]")
    print(f"    RMSE: {m.rmse:.2%}")
    print(f"    Spearman r: {m.spearman_r:+.3f} (p={m.spearman_p:.3f})" if m.spearman_r is not None
          else "    Spearman r: n/a (not enough variation)")
    print(f"    Brier score: {m.brier_score:.3f} (lower is better; 0.25 = coin flip at 50% confidence)")


def main() -> None:
    clock = PointInTimeClock(now=datetime.now(timezone.utc))
    print(f"[1/4] Fetching real SEC EDGAR guidance-change events, {START_DATE} to {END_DATE} "
          f"(cached at {CACHE_PATH})...")
    sourced = fetch_guidance_events(START_DATE, END_DATE, clock, cache_path=CACHE_PATH, max_pages_per_phrase=10)
    print(f"  {len(sourced)} real, deduplicated events.")

    entities = sorted({s.raw_item.entity for s in sourced})
    print(f"\n[2/4] Loading real Yahoo Finance price history for {len(entities)} entities + SPY "
          "(cached to data_cache/prices/)...")
    prices = YahooPriceSeriesProvider(cache_dir="data_cache/prices")
    prices._load("SPY")
    n_ok = 0
    for ticker in entities:
        try:
            n_ok += 0 if prices._load(ticker).empty else 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  {n_ok}/{len(entities)} entities have usable price history.")

    print(f"\n[3/4] Running the chronological walk-forward Static-vs-Adaptive experiment, "
          f"horizons={HORIZONS}...")
    config = WalkforwardConfig(horizon_days_list=HORIZONS, embargo_days=2, benchmark_ticker="SPY",
                                burn_in_fraction=0.20, final_holdout_fraction=0.20)
    conn = db.connect("data_cache/real_experiment.sqlite")
    report = run_walkforward(sourced, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)

    print("\n[4/4] Building knowledge-state and chronological before/after reports...")
    knowledge = build_knowledge_state_report(conn, now=datetime.now(timezone.utc))
    go_live = [(r["relationship_id"], datetime.fromisoformat(r["shadow_promoted_at"]))
               for r in conn.execute(
                   "SELECT relationship_id, shadow_promoted_at FROM validated_relationships "
                   "WHERE shadow_promoted_at IS NOT NULL").fetchall()]
    chrono = evaluate_chronologically(report.scored, go_live)

    # ============================== REPORT ==============================
    print("\n" + "=" * 78)
    print("REAL-DATA WALK-FORWARD EXPERIMENT REPORT - STAGE 4")
    print("=" * 78)
    for line in report.evidence:
        print(f"  {line}")

    n_hyp_generated = len(report.promotions) + len(report.rejections)
    n_promoted_shadow = len(report.promotions)
    n_rejected = len(report.rejections)
    n_went_live = len(go_live)
    n_retired = sum(1 for r in report.revalidations if r.get("new_status") == "RETIRED") + \
        sum(1 for s in report.shadow_evaluations if s["outcome"] == "RETIRED")

    print(f"\n  Real events processed:        {len(sourced)}")
    print(f"  Predictions generated:        {len(report.scored)} ({len(report.scored)//2} events x "
          f"{len(HORIZONS)} horizons x 2 agents)")
    print(f"  Outcomes resolved:             {sum(1 for s in report.scored if s.realized_abnormal_return is not None)}")
    print(f"  Hypotheses generated & tested: {n_hyp_generated}")
    print(f"  Hypotheses entered shadow:     {n_promoted_shadow}")
    print(f"  Hypotheses rejected:           {n_rejected}")
    print(f"  Relationships that went LIVE (SHADOW->ACTIVE): {n_went_live}")
    print(f"  Relationships retired:         {n_retired}")
    print(f"  Final knowledge version:       {knowledge.knowledge_version}")

    for horizon in HORIZONS:
        print(f"\n{'=' * 78}\nHORIZON: {horizon}D\n{'=' * 78}")
        for segment in ("DEVELOPMENT", "FINAL_HOLDOUT"):
            print(f"\n--- {segment} ---")
            for agent in ("STATIC", "ADAPTIVE"):
                rows = [s for s in report.scored if s.segment == segment and s.agent == agent
                        and s.horizon_days == horizon]
                outcomes = [PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return)
                            for s in rows if s.realized_abnormal_return is not None]
                _print_metrics_block(f"{agent} ({segment}, {horizon}D)", outcomes)

    print(f"\n{'=' * 78}\nCHRONOLOGICAL BEFORE/AFTER EVALUATION (all horizons pooled)\n{'=' * 78}")
    for line in chrono.evidence:
        print(f"  {line}")
    for w in chrono.windows:
        label = "BEFORE any update" if w.update_relationship_id is None else f"AFTER update ({w.update_relationship_id})"
        print(f"\n  Window {w.window_index} [{label}]: N={w.n_predictions}")
        if w.adaptive_mae_improved_vs_static is not None:
            print(f"    ADAPTIVE MAE {'IMPROVED' if w.adaptive_mae_improved_vs_static else 'did NOT improve'} "
                  f"vs. STATIC in this window.")
        if w.adaptive_direction_improved_vs_static is not None:
            print(f"    ADAPTIVE direction accuracy "
                  f"{'IMPROVED' if w.adaptive_direction_improved_vs_static else 'did NOT improve'} vs. STATIC.")

    print(f"\n{'=' * 78}\nKNOWLEDGE STATE\n{'=' * 78}")
    print(f"  ACTIVE relationships: {len(knowledge.active_relationships)}")
    for r in knowledge.active_relationships:
        print(f"    {r.relationship_id}: {r.condition} -> {r.effect_estimate:+.2%} (N={r.n_supporting}, "
              f"CI=[{r.ci_low}, {r.ci_high}], decay={r.decay_state}, supported={r.n_predictions_supported}, "
              f"contradicted={r.n_predictions_contradicted})")
    print(f"  SHADOW relationships (probation, not yet live): {len(knowledge.shadow_relationships)}")
    for r in knowledge.shadow_relationships:
        print(f"    {r.relationship_id}: {r.condition} -> {r.effect_estimate:+.2%} (N={r.n_supporting})")
    print(f"  RETIRED relationships: {len(knowledge.retired_relationships)}")
    print(f"  Rejected hypotheses: {len(knowledge.rejected_hypotheses)}")
    reasons = Counter(h["reason"] for h in knowledge.rejected_hypotheses)
    print(f"    Rejection reasons: {dict(reasons)}")
    print(f"  Source reliability:")
    for s in knowledge.source_reliability:
        hr = f"{s.hit_rate:.1%}" if s.hit_rate is not None else "n/a"
        print(f"    {s.source}: N={s.n_resolved_predictions}, hit_rate={hr}")

    print("\n" + "=" * 78)
    print(f"Configuration was fixed before this run. Results above are reported as obtained, per horizon, "
          f"per segment, with no post-hoc adjustment of MIN_N, ALPHA, embargo, shadow-probation size, or "
          f"burn-in/holdout fractions based on what this run showed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
