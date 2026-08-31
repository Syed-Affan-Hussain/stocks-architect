"""Stage 7 item 10 - THE FINAL REAL-DATA RUN: the complete pipeline (five-
way walk-forward with frozen TRAIN/VALIDATE/SHADOW/TEST discipline,
hierarchical research, incremental-value/permutation/temporal-stability
diagnostics, StrategyAgent economics, strategy-level TEST evaluation) run
ONCE against the existing real cached SEC EDGAR guidance events + Yahoo
Finance OHLCV, with the six-state final report (reporting/
stage7_final_report.py) printed and saved.

    python scripts/run_stage7_final_report.py

NO POST-HOC TUNING: every threshold (MIN_N, ALPHA, MIN_ECONOMIC_EFFECT,
ResearchBudget, COST_MARGIN_MULTIPLE, MIN_ECONOMIC_TRADES) is exactly what
earlier stage-7 items already fixed and committed BEFORE this script ever
ran against real data. This script does not retune anything based on what
it finds, and reports a negative or empty result exactly as honestly as a
positive one - see stage 7's own explicit instruction: "if nothing
survives, that is a valid scientific result."

ONE HIERARCHICAL-RESEARCH PASS PER DIRECTION/HORIZON, REUSED TWICE: the
SAME set of HierarchicalResearchReport objects (built by
`_compute_qualified_relationships`, which the five-way walk-forward calls
exactly once at the VALIDATE/TEST boundary via
`compute_qualified_relationships_fn` - see experiment/four_way_walkforward.py)
is reused directly by build_stage7_final_report afterward. Running
hierarchical research a second, separate time would be wasteful AND would
risk a second pass silently seeing different state; reusing the exact
objects the ENSEMBLE_ADAPTIVE reconfiguration itself was built from
guarantees both consumers see identical evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.four_way_walkforward import AGENT_NAMES, FourWayWalkforwardConfig, run_four_way_walkforward
from market_agent.experiment.metrics import PredictionOutcome, compute_metrics
from market_agent.learn.hierarchical_research import DEFAULT_RESEARCH_BUDGET, HierarchicalResearchReport, run_hierarchical_research_pass
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.pit.clock import PointInTimeClock
from market_agent.reporting.stage7_final_report import build_stage7_final_report
from market_agent.sources.edgar_guidance import fetch_guidance_events
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
GUIDANCE_CACHE = "data_cache/edgar/guidance_2016_2024.json"
LEDGER_PATH = "data_cache/stage7_final_report.sqlite"
REPORT_JSON_PATH = "data_cache/stage7_final_report.json"
REPORT_TEXT_PATH = "data_cache/stage7_final_report.txt"
HORIZONS = [1, 5, 20, 60]


def _find_relationship_id(conn, condition: dict, horizon_days: int) -> str | None:
    for row in conn.execute("SELECT relationship_id, condition_json FROM validated_relationships "
                             "WHERE horizon_days = ?", (horizon_days,)).fetchall():
        if json.loads(row["condition_json"]) == condition:
            return row["relationship_id"]
    return None


def _shadow_worthy(result) -> bool:
    """A Level 2/3 result that clears ALL FOUR checks (baseline-scalar
    significance, incremental value, permutation, temporal stability) -
    exactly the SHADOW state in reporting/stage7_final_report.py's
    six-state taxonomy. Used to decide ENSEMBLE_ADAPTIVE's qualified set,
    independently of the final report's own (identical) walk of the same
    evidence hierarchy."""
    return (result.test_result.status == "CONFIRMED"
            and result.incremental_value is not None
            and result.incremental_value.status == "INCREMENTAL_VALUE_CONFIRMED"
            and result.permutation_test is not None
            and result.permutation_test.status == "SURVIVES_PERMUTATION"
            and result.temporal_stability is not None
            and result.temporal_stability.status == "STABLE_ACROSS_TIME")


def _make_compute_qualified(research_reports: list[HierarchicalResearchReport]):
    def compute_qualified(conn, test_boundary_iso: str, unconditional_baseline: dict) -> set[str]:
        print(f"\n  [ENSEMBLE_ADAPTIVE reconfiguration] Running hierarchical research pinned at the VALIDATE/TEST "
              f"boundary ({test_boundary_iso})...")
        qualified: set[str] = set()
        for direction in ("positive", "negative"):
            for horizon in HORIZONS:
                research = run_hierarchical_research_pass(
                    conn, "GUIDANCE_CHANGE", direction, horizon, test_boundary_iso, unconditional_baseline,
                    proposed_at=datetime.fromisoformat(test_boundary_iso), promoted_by="stage7-final-report",
                    budget=DEFAULT_RESEARCH_BUDGET)
                research_reports.append(research)
                for results in research.level2_results.values():
                    for r in results:
                        if _shadow_worthy(r):
                            rel_id = _find_relationship_id(conn, r.condition, horizon)
                            if rel_id:
                                qualified.add(rel_id)
                for results in research.level3_results.values():
                    for r in results:
                        if _shadow_worthy(r):
                            rel_id = _find_relationship_id(conn, r.condition, horizon)
                            if rel_id:
                                qualified.add(rel_id)
        print(f"  [ENSEMBLE_ADAPTIVE reconfiguration] {len(qualified)} SHADOW-worthy relationship(s) qualified "
              f"out of {sum(len(r.level2_results) + len(r.level3_results) for r in research_reports)} "
              "dimension/setup groups screened.")
        return qualified
    return compute_qualified


def _five_way_summary(report) -> dict:
    """Compares each ADAPTIVE variant against STATIC on FINAL_HOLDOUT
    (TEST) segment predictions only, via MAE (experiment/metrics.py) -
    lower MAE is better. This is the ONLY segment used for this
    comparison, so "outperforms" here already means "on the frozen TEST
    segment", answering both five-way questions (item 9's 7th and 8th) at
    once."""
    holdout = [s for s in report.scored if s.segment == "FINAL_HOLDOUT"]
    by_agent: dict[str, list] = {}
    for s in holdout:
        if s.predicted_impact is None or s.realized_abnormal_return is None:
            continue
        by_agent.setdefault(s.agent, []).append(
            PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return))

    metrics_by_agent = {agent: compute_metrics(outcomes) for agent, outcomes in by_agent.items() if outcomes}
    static_mae = metrics_by_agent.get("STATIC").mae if "STATIC" in metrics_by_agent else None

    lines = []
    better_than_static = []
    for agent in AGENT_NAMES:
        if agent == "STATIC" or agent not in metrics_by_agent:
            continue
        m = metrics_by_agent[agent]
        lines.append(f"{agent}: MAE={m.mae!r} (N={m.n}) vs STATIC MAE={static_mae!r}")
        if static_mae is not None and m.mae is not None and m.mae < static_mae:
            better_than_static.append(agent)

    if static_mae is None:
        answer = "No STATIC TEST-segment predictions with a resolved outcome to compare against."
    elif better_than_static:
        answer = (f"Yes, on the frozen TEST segment, by mean absolute error: {', '.join(better_than_static)} "
                   f"outperform STATIC (MAE={static_mae:.4f}). " + "; ".join(lines))
    else:
        answer = f"No - no ADAPTIVE variant beats STATIC's TEST-segment MAE ({static_mae:.4f}). " + "; ".join(lines)

    return {"does_adaptive_outperform_static": answer, "metrics_by_agent": {a: m.mae for a, m in metrics_by_agent.items()}}


def main() -> None:
    clock = PointInTimeClock(now=datetime.now(timezone.utc))
    Path("data_cache").mkdir(exist_ok=True)
    ledger_path = Path(LEDGER_PATH)
    if ledger_path.exists():
        ledger_path.unlink()
    conn = db.connect(str(ledger_path))

    print(f"[1/4] Loading cached real SEC EDGAR guidance events, {START_DATE} to {END_DATE}...")
    guidance_sourced = fetch_guidance_events(START_DATE, END_DATE, clock, cache_path=GUIDANCE_CACHE,
                                              max_pages_per_phrase=10)
    print(f"  {len(guidance_sourced)} events.")

    entities = sorted({s.raw_item.entity for s in guidance_sourced})
    print(f"\n[2/4] Loading real Yahoo Finance OHLCV for {len(entities)} entities + SPY...")
    prices = YahooPriceSeriesProvider(cache_dir="data_cache/prices")
    prices._load_frame("SPY")
    n_ok = 0
    for ticker in entities:
        try:
            n_ok += 0 if prices._load_frame(ticker).empty else 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  {n_ok}/{len(entities)} entities have usable OHLCV history.")

    print("\n[3/4] Running the five-way walk-forward (STATIC / CURRENT_ADAPTIVE / TECHNICAL_ADAPTIVE / "
          "METHODOLOGY_ADAPTIVE / ENSEMBLE_ADAPTIVE), TRAIN/VALIDATE/SHADOW/TEST discipline "
          "(freeze_governance_during_test=True), ENSEMBLE_ADAPTIVE reconfigured once at the boundary from a "
          "real hierarchical-research pass...")
    config = FourWayWalkforwardConfig(horizon_days_list=HORIZONS, embargo_days=2, benchmark_ticker="SPY",
                                       burn_in_fraction=0.20, final_holdout_fraction=0.20,
                                       freeze_governance_during_test=True)
    reactive_generator = RuleBasedHypothesisGenerator(include_technical_dimensions=False)
    research_reports: list[HierarchicalResearchReport] = []
    compute_qualified = _make_compute_qualified(research_reports)

    report = run_four_way_walkforward(guidance_sourced, prices, RuleBasedInterpreter(), reactive_generator, config,
                                       conn, ohlcv=prices, compute_qualified_relationships_fn=compute_qualified)

    for line in report.evidence:
        print(f"  {line}")

    dev_predictions = [s for s in report.scored if s.segment == "DEVELOPMENT"]
    holdout_predictions = [s for s in report.scored if s.segment == "FINAL_HOLDOUT"]
    boundary = max(s.published_at for s in dev_predictions)
    print(f"\n  VALIDATE/TEST boundary: {boundary.isoformat()} "
          f"({len(dev_predictions)} VALIDATE-segment scored predictions before it, "
          f"{len(holdout_predictions)} TEST-segment predictions after)")

    five_way_summary = _five_way_summary(report)
    print(f"\n  {five_way_summary['does_adaptive_outperform_static']}")

    print(f"\n[4/4] Building the stage-7 final report from the {len(research_reports)} hierarchical-research "
          "pass(es) already computed above (reused, not re-run)...")
    final_report = build_stage7_final_report(conn, prices, research_reports, report.unconditional_baseline,
                                              boundary.isoformat(), five_way_summary=five_way_summary)

    text = final_report.to_text()
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")

    Path(REPORT_TEXT_PATH).write_text(text, encoding="utf-8")
    Path(REPORT_JSON_PATH).write_text(json.dumps(final_report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {REPORT_TEXT_PATH}\nSaved: {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
