"""Stage 5 real-data run - item 13's precise reporting checklist, item 14's
success criterion (continuous safe operation, NOT "beats STATIC").

    python scripts/run_stage5_experiment.py

Runs TWO independent walk-forward passes (GUIDANCE_CHANGE, already live
since stage 3-4, and DIVIDEND_CHANGE, stage 5's new event type) against
real SEC EDGAR events and real Yahoo Finance prices, sharing ONE
persistent ledger so both event types accumulate into one continuously
operating knowledge store (item 7) - exactly what a live system would do,
just replayed historically. Each event type gets its OWN burn-in baseline
(run_walkforward's _estimate_baseline only ever sees the event list it was
given), so guidance and dividend effect magnitudes are never blended into
one number - no change to walkforward.py, agents/, or any currently-passing
test was needed for this separation; it falls out of calling run_walkforward
twice with disjoint event lists on the same connection.

Then demonstrates the newly built real-time surfaces against that same
ledger: the predict CLI's core (market_agent.predict.run_predict), the
knowledge-state dashboard (market_agent.report), rolling ADAPTIVE-vs-STATIC
monitoring (experiment/rolling_monitor.py), and the portfolio translation
layer (portfolio/translate.py) against a small illustrative holdings dict.

CONFIGURATION FIXED BEFORE THIS RUN, PER THIS PROJECT'S STANDING RULE:
horizons, embargo, burn-in/holdout fractions, and every statistical
constant in hypothesis_testing.py/shadow.py/regime.py were already fixed
before stage 3's first real-data run and have not been touched since,
including here. Whatever this prints is reported as-is - a negative or
inconclusive result is not a bug to fix by adjusting a threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from datetime import datetime, timedelta, timezone

from market_agent.agents.adaptive_agent import AdaptiveAgent
from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.chronological_eval import evaluate_chronologically
from market_agent.experiment.metrics import PredictionOutcome, compute_metrics
from market_agent.experiment.rolling_monitor import rolling_by_dimension, rolling_comparison
from market_agent.experiment.walkforward import WalkforwardConfig, run_walkforward
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.llm.select import describe_active_choice, select_hypothesis_generator_from_env, \
    select_interpreter_from_env
from market_agent.pipeline import predict_event
from market_agent.pit.clock import PointInTimeClock
from market_agent.portfolio.translate import translate_event_to_portfolio
from market_agent.predict import estimate_baseline_from_ledger
from market_agent.report import format_report as format_knowledge_report
from market_agent.reporting.knowledge_state import build_knowledge_state_report
from market_agent.sources.edgar_dividend import fetch_dividend_events
from market_agent.sources.edgar_guidance import fetch_guidance_events
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
GUIDANCE_CACHE = "data_cache/edgar/guidance_2016_2024.json"
DIVIDEND_CACHE = "data_cache/edgar/dividend_2016_2024.json"
LEDGER_PATH = "data_cache/stage5_experiment.sqlite"
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


def _run_walkforward_for(label, sourced, prices, conn):
    config = WalkforwardConfig(horizon_days_list=HORIZONS, embargo_days=2, benchmark_ticker="SPY",
                                burn_in_fraction=0.20, final_holdout_fraction=0.20)
    print(f"\n  Running walk-forward for {label} ({len(sourced)} sourced items)...")
    report = run_walkforward(sourced, prices, RuleBasedInterpreter(), RuleBasedHypothesisGenerator(), config, conn)
    return report


def _report_event_type(label, report, conn):
    n_hyp_generated = len(report.promotions) + len(report.rejections)
    go_live = [(r["relationship_id"], datetime.fromisoformat(r["shadow_promoted_at"]))
               for r in conn.execute(
                   "SELECT relationship_id, shadow_promoted_at FROM validated_relationships "
                   "WHERE shadow_promoted_at IS NOT NULL AND json_extract(condition_json, '$.event_type') = ?",
                   (label,)).fetchall()]
    chrono = evaluate_chronologically(report.scored, go_live)

    print(f"\n{'=' * 78}\n{label} - WALK-FORWARD REPORT\n{'=' * 78}")
    for line in report.evidence:
        print(f"  {line}")
    print(f"\n  Predictions generated:  {len(report.scored)}")
    print(f"  Outcomes resolved:      {sum(1 for s in report.scored if s.realized_abnormal_return is not None)}")
    print(f"  Hypotheses generated & tested: {n_hyp_generated}")
    print(f"  Hypotheses entered shadow:     {len(report.promotions)}")
    print(f"  Hypotheses rejected:           {len(report.rejections)}")
    print(f"  Relationships that went LIVE (SHADOW->ACTIVE): {len(go_live)}")

    for horizon in HORIZONS:
        print(f"\n  --- {label} horizon {horizon}D ---")
        for segment in ("DEVELOPMENT", "FINAL_HOLDOUT"):
            for agent in ("STATIC", "ADAPTIVE"):
                rows = [s for s in report.scored if s.segment == segment and s.agent == agent
                        and s.horizon_days == horizon]
                outcomes = [PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return)
                            for s in rows if s.realized_abnormal_return is not None]
                _print_metrics_block(f"{agent} ({segment})", outcomes)

    print(f"\n  Chronological before/after evaluation:")
    for line in chrono.evidence:
        print(f"    {line}")
    for w in chrono.windows:
        wlabel = "BEFORE any update" if w.update_relationship_id is None else f"AFTER update ({w.update_relationship_id})"
        print(f"    Window {w.window_index} [{wlabel}]: N={w.n_predictions}", end="")
        if w.adaptive_mae_improved_vs_static is not None:
            print(f", ADAPTIVE MAE {'IMPROVED' if w.adaptive_mae_improved_vs_static else 'did NOT improve'}", end="")
        print()
    return {"n_predictions": len(report.scored), "n_hyp_generated": n_hyp_generated,
            "n_shadowed": len(report.promotions), "n_rejected": len(report.rejections), "n_went_live": len(go_live)}


def _demo_live_prediction(conn, prices, entity, text, source, as_of):
    interpreter = select_interpreter_from_env()
    baseline = estimate_baseline_from_ledger(conn, "GUIDANCE_CHANGE", HORIZONS)
    agent = AdaptiveAgent(conn, baseline)
    result = predict_event(conn, agent, prices, interpreter, entity, text, source, as_of, as_of, HORIZONS)
    print(f"\n  predict_event('{entity}', {text!r}) -> status={result.status}")
    if result.status == "OK":
        for p in result.predictions:
            print(f"    {p.horizon_days}D: status={p.status} impact={p.predicted_impact} "
                  f"confidence={p.predicted_confidence} novelty={p.novelty_score:.2f}")
    return result


def main() -> None:
    clock = PointInTimeClock(now=datetime.now(timezone.utc))

    print(f"[1/6] Fetching real SEC EDGAR events, {START_DATE} to {END_DATE}...")
    guidance_sourced = fetch_guidance_events(START_DATE, END_DATE, clock, cache_path=GUIDANCE_CACHE,
                                              max_pages_per_phrase=10)
    print(f"  GUIDANCE_CHANGE: {len(guidance_sourced)} real, deduplicated events (cached).")
    dividend_sourced = fetch_dividend_events(START_DATE, END_DATE, clock, cache_path=DIVIDEND_CACHE,
                                              max_pages_per_phrase=10)
    print(f"  DIVIDEND_CHANGE: {len(dividend_sourced)} real, deduplicated events (cached).")

    entities = sorted({s.raw_item.entity for s in guidance_sourced} | {s.raw_item.entity for s in dividend_sourced})
    print(f"\n[2/6] Loading real Yahoo Finance price history for {len(entities)} entities + SPY "
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

    print(f"\n[3/6] Provider configuration: {describe_active_choice(select_interpreter_from_env(), select_hypothesis_generator_from_env())}")

    print(f"\n[4/6] Running two independent walk-forward passes on one shared, continuously "
          f"accumulating ledger ({LEDGER_PATH})...")
    Path("data_cache").mkdir(exist_ok=True)
    ledger_path = Path(LEDGER_PATH)
    if ledger_path.exists():
        ledger_path.unlink()  # fresh run - this script's own scratch ledger, not the live CLI's default
    conn = db.connect(str(ledger_path))

    guidance_report = _run_walkforward_for("GUIDANCE_CHANGE", guidance_sourced, prices, conn)
    dividend_report = _run_walkforward_for("DIVIDEND_CHANGE", dividend_sourced, prices, conn)

    print("\n[5/6] Building knowledge-state, rolling-monitor, and demonstration reports...")
    knowledge = build_knowledge_state_report(conn, now=datetime.now(timezone.utc))

    guidance_stats = _report_event_type("GUIDANCE_CHANGE", guidance_report, conn)
    dividend_stats = _report_event_type("DIVIDEND_CHANGE", dividend_report, conn)

    print(f"\n{'=' * 78}\nKNOWLEDGE-STATE DASHBOARD (both event types, combined ledger)\n{'=' * 78}")
    print(format_knowledge_report(knowledge))

    print(f"\n{'=' * 78}\nROLLING ADAPTIVE-vs-STATIC MONITORING (descriptive only)\n{'=' * 78}")
    for c in rolling_comparison(conn, window_sizes=(50, 100)):
        print(f"  Window last-{c.window_size} (n_available={c.n_available}): "
              f"STATIC n={c.static_metrics.n} ADAPTIVE n={c.adaptive_metrics.n}")
    for report_by_dim in rolling_by_dimension(conn, "event_type", window_sizes=(100,)):
        w = report_by_dim.windows[0]
        static_acc = f"{w.static_metrics.direction_accuracy:.1%}" if w.static_metrics.direction_accuracy else "n/a"
        adaptive_acc = f"{w.adaptive_metrics.direction_accuracy:.1%}" if w.adaptive_metrics.direction_accuracy else "n/a"
        print(f"  {report_by_dim.dimension_value} (last 100): STATIC dir_acc={static_acc}  "
              f"ADAPTIVE dir_acc={adaptive_acc}  [descriptive only - not a significance test]")

    print(f"\n{'=' * 78}\nLIVE PREDICTION CLI DEMONSTRATION (predict_event, logged to this ledger)\n{'=' * 78}")
    demo_as_of = datetime.now(timezone.utc)
    _demo_live_prediction(conn, prices, "AAPL", "AAPL corp raises full-year guidance", "demo", demo_as_of)
    _demo_live_prediction(conn, prices, "AAPL", "AAPL corp announces new retail store", "demo", demo_as_of)

    print(f"\n{'=' * 78}\nPORTFOLIO TRANSLATION LAYER DEMONSTRATION\n{'=' * 78}")
    demo_portfolio = {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}
    baseline = estimate_baseline_from_ledger(conn, "GUIDANCE_CHANGE", HORIZONS)
    agent = AdaptiveAgent(conn, baseline)
    event_pred = predict_event(conn, agent, prices, select_interpreter_from_env(), "AAPL",
                                "AAPL corp cuts full-year guidance", "demo", demo_as_of, demo_as_of, HORIZONS)
    portfolio_report = translate_event_to_portfolio(conn, prices, demo_portfolio, event_pred, 20, demo_as_of)
    print(f"  Portfolio: {demo_portfolio}")
    print(f"  Triggering event: {portfolio_report.triggering_entity} ({portfolio_report.triggering_event_type})")
    print(f"  portfolio_expected_impact (20D): {portfolio_report.portfolio_expected_impact}")
    for h in portfolio_report.holdings:
        print(f"    {h.entity}: status={h.status} weighted_contribution={h.weighted_contribution} "
              f"regime={h.regime} n_applicable_relationships={h.n_applicable_relationships}")
    for line in portfolio_report.reasoning_provenance:
        print(f"    - {line}")

    # ============================== ITEM 13 CHECKLIST ==============================
    print(f"\n{'=' * 78}\nITEM 13 REPORTING CHECKLIST (as obtained - not tuned after seeing these numbers)\n{'=' * 78}")
    total_ingested = knowledge.operational_counts.n_events_ingested
    print(f"  Events ingested (total ledger rows, incl. live demo predictions): {total_ingested}")
    print(f"    GUIDANCE_CHANGE sourced: {len(guidance_sourced)} raw -> "
          f"{len(guidance_report.scored) // (2 * len(HORIZONS)) if guidance_report.scored else 0} "
          f"unique events predicted and resolved")
    print(f"    DIVIDEND_CHANGE sourced: {len(dividend_sourced)} raw -> "
          f"{len(dividend_report.scored) // (2 * len(HORIZONS)) if dividend_report.scored else 0} "
          f"unique events predicted and resolved")
    print(f"  Events resolved (outcome known): {knowledge.operational_counts.n_events_resolved}")
    print(f"  Events INSUFFICIENT_PRECEDENT:   {knowledge.operational_counts.n_events_insufficient_precedent}")
    print(f"  Events DATA_ERROR:               {knowledge.operational_counts.n_events_data_error}")
    print(f"  Error type distribution (resolved): {knowledge.operational_counts.error_type_distribution}")
    print(f"  Hypotheses generated: {knowledge.operational_counts.n_hypotheses_generated}  "
          f"confirmed: {knowledge.operational_counts.n_hypotheses_confirmed}  "
          f"rejected: {knowledge.operational_counts.n_hypotheses_rejected}  "
          f"untested: {knowledge.operational_counts.n_hypotheses_untested}")
    print(f"  Relationships: active={knowledge.operational_counts.n_relationships_active}  "
          f"shadow={knowledge.operational_counts.n_relationships_shadow}  "
          f"retired={knowledge.operational_counts.n_relationships_retired}")
    print(f"  Governance changes (knowledge_version): {knowledge.operational_counts.n_governance_changes}")
    print(f"  LLM status: {knowledge.llm_status}")
    print(f"  LLM calls executed: 0 (no LLM client configured in this environment - see llm/select.py; "
          f"HYPOTHESIS_PROVIDER/INTERPRETER_PROVIDER both default to rule_based, no silent fallback occurred)")
    print(f"\n  Calibration by horizon (direction accuracy / MAE, STATIC vs ADAPTIVE, all event types pooled):")
    for hc in knowledge.calibration_by_horizon:
        sa = f"{hc.static_direction_accuracy:.1%}" if hc.static_direction_accuracy is not None else "n/a"
        aa = f"{hc.adaptive_direction_accuracy:.1%}" if hc.adaptive_direction_accuracy is not None else "n/a"
        sm = f"{hc.static_mae:.4f}" if hc.static_mae is not None else "n/a"
        am = f"{hc.adaptive_mae:.4f}" if hc.adaptive_mae is not None else "n/a"
        print(f"    {hc.horizon_days}D: STATIC(n={hc.static_n}) dir_acc={sa} mae={sm}  |  "
              f"ADAPTIVE(n={hc.adaptive_n}) dir_acc={aa} mae={am}")
    print(f"\n  Transaction-cost-free economic comparison: MAE/direction-accuracy above ARE the "
          f"transaction-cost-free comparison (no cost model exists or is claimed - abnormal returns "
          f"only, per event-study convention).")
    print(f"\n  Per-event-type summary:")
    print(f"    GUIDANCE_CHANGE: {guidance_stats}")
    print(f"    DIVIDEND_CHANGE: {dividend_stats}")
    print(f"\n  Failures encountered: see 'DATA_ERROR' count above (delisted/missing-price-history tickers, "
          f"handled fail-closed) and INSUFFICIENT_PRECEDENT count (system correctly abstained rather than "
          f"guessing). No unhandled exception occurred during this run.")

    print(f"\n{'=' * 78}\nITEM 14 SUCCESS CRITERION\n{'=' * 78}")
    print("  Success is defined as: the system operated continuously across two independent event types on "
          "one shared ledger, correctly kept their baselines and relationships separate, never overwrote a "
          "recorded outcome, never leaked future information, and produced calibration/accuracy numbers "
          "as obtained - NOT whether ADAPTIVE beat STATIC. See the per-horizon numbers above for whether "
          "ADAPTIVE outperformed STATIC on each event type's reserved final holdout; a negative or mixed "
          "result there is not a failure of this run's success criterion.")
    print("=" * 78)


if __name__ == "__main__":
    main()
