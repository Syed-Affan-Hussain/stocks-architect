"""The definitive stage-7 real-data report: hierarchical, budget-bounded
research (learn/hierarchical_research.py) + incremental-value testing
(learn/incremental_value.py) + anti-overfitting diagnostics
(learn/overfitting_diagnostics.py), run with the TRAIN/VALIDATE/SHADOW/TEST
discipline (experiment/four_way_walkforward.py's
freeze_governance_during_test=True) so the final holdout is genuinely
frozen for governance, not just separately reported.

    python scripts/validate_stage7_hierarchical_research.py

Reuses stage 5/6's cached real EDGAR guidance events and Yahoo OHLCV -
nothing new is fetched. Runs the four-way walk-forward harness with the
REACTIVE hypothesis generator set to include_technical_dimensions=False
(so CURRENT_ADAPTIVE/EVENT_ADAPTIVE's mechanism is exactly stage 1-5's
event-context-only path, unchanged) and freeze_governance_during_test=True
(TEST/final-holdout predictions still happen, but no new hypothesis,
promotion, or shadow evaluation is recorded once chronological time
crosses into it). The hierarchical research pass then runs SEPARATELY,
with `published_before` fixed at the VALIDATE/TEST boundary - it never
sees final-holdout data, matching the harness's own frozen discipline.

CONFIGURATION FIXED BEFORE THIS RUN: ResearchBudget, MIN_N, ALPHA,
MIN_ECONOMIC_EFFECT, N_PERMUTATIONS are exactly what earlier stages
already fixed. This script does not retune anything based on what it
finds.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.four_way_walkforward import FourWayWalkforwardConfig, run_four_way_walkforward
from market_agent.learn.hierarchical_research import DEFAULT_RESEARCH_BUDGET, run_hierarchical_research_pass
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.pit.clock import PointInTimeClock
from market_agent.sources.edgar_guidance import fetch_guidance_events
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
GUIDANCE_CACHE = "data_cache/edgar/guidance_2016_2024.json"
LEDGER_PATH = "data_cache/stage7_validation.sqlite"
HORIZONS = [1, 5, 20, 60]


def _print_confirmed_detail(dimension_or_setup: str, level_label: str, results) -> None:
    for r in results:
        if r.test_result.status != "CONFIRMED":
            continue
        iv = r.incremental_value
        pt = r.permutation_test
        ts = r.temporal_stability
        iv_s = iv.status if iv else "n/a"
        pt_s = pt.status if pt else "n/a"
        ts_s = ts.status if ts else "n/a"
        print(f"      [{level_label}] {dimension_or_setup}: effect={r.test_result.mean_effect:+.2%} "
              f"N={r.test_result.n}  incremental={iv_s}  permutation={pt_s}  stability={ts_s}")


def main() -> None:
    clock = PointInTimeClock(now=datetime.now(timezone.utc))
    Path("data_cache").mkdir(exist_ok=True)
    ledger_path = Path(LEDGER_PATH)
    if ledger_path.exists():
        ledger_path.unlink()
    conn = db.connect(str(ledger_path))

    print(f"[1/3] Loading cached real SEC EDGAR guidance events, {START_DATE} to {END_DATE}...")
    guidance_sourced = fetch_guidance_events(START_DATE, END_DATE, clock, cache_path=GUIDANCE_CACHE,
                                              max_pages_per_phrase=10)
    print(f"  {len(guidance_sourced)} events.")

    entities = sorted({s.raw_item.entity for s in guidance_sourced})
    print(f"\n[2/3] Loading real Yahoo Finance OHLCV for {len(entities)} entities + SPY...")
    prices = YahooPriceSeriesProvider(cache_dir="data_cache/prices")
    prices._load_frame("SPY")
    n_ok = 0
    for ticker in entities:
        try:
            n_ok += 0 if prices._load_frame(ticker).empty else 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  {n_ok}/{len(entities)} entities have usable OHLCV history.")

    print("\n[3/3] Running the walk-forward pass (event-context-only reactive path, TRAIN/VALIDATE/SHADOW/TEST "
          "discipline: freeze_governance_during_test=True), then the hierarchical research pass on "
          "VALIDATE-segment data only...")
    config = FourWayWalkforwardConfig(horizon_days_list=HORIZONS, embargo_days=2, benchmark_ticker="SPY",
                                       burn_in_fraction=0.20, final_holdout_fraction=0.20,
                                       freeze_governance_during_test=True)
    reactive_generator = RuleBasedHypothesisGenerator(include_technical_dimensions=False)
    report = run_four_way_walkforward(guidance_sourced, prices, RuleBasedInterpreter(), reactive_generator,
                                       config, conn, ohlcv=prices)

    for line in report.evidence:
        print(f"  {line}")

    dev_predictions = [s for s in report.scored if s.segment == "DEVELOPMENT"]
    holdout_predictions = [s for s in report.scored if s.segment == "FINAL_HOLDOUT"]
    boundary = max(s.published_at for s in dev_predictions)
    test_start = min(s.published_at for s in holdout_predictions) if holdout_predictions else None
    print(f"\n  VALIDATE/TEST boundary: {boundary.isoformat()} "
          f"({len(dev_predictions)} VALIDATE-segment scored predictions before it, "
          f"{len(holdout_predictions)} TEST-segment predictions after)")

    if test_start is not None:
        n_governance_after_test = conn.execute(
            "SELECT COUNT(*) c FROM model_registry WHERE created_at >= ?", (test_start.isoformat(),)
        ).fetchone()["c"]
        print(f"  Governance actions recorded at/after the TEST boundary: {n_governance_after_test} "
              "(must be 0 - freeze_governance_during_test=True)")

    print(f"\n{'=' * 78}\nHIERARCHICAL RESEARCH PASS RESULTS (VALIDATE-segment data only)\n{'=' * 78}")
    for direction in ("positive", "negative"):
        for horizon in HORIZONS:
            research = run_hierarchical_research_pass(
                conn, "GUIDANCE_CHANGE", direction, horizon, boundary.isoformat(),
                report.unconditional_baseline, proposed_at=boundary, promoted_by="stage7-validation",
                budget=DEFAULT_RESEARCH_BUDGET)
            n_confirmed_l1 = sum(1 for r in research.level1_results if r.test_result.status == "CONFIRMED")
            n_confirmed_l2 = sum(1 for results in research.level2_results.values()
                                  for r in results if r.test_result.status == "CONFIRMED")
            n_confirmed_l2_incremental = sum(1 for results in research.level2_results.values() for r in results
                                              if r.test_result.status == "CONFIRMED" and r.incremental_value
                                              and r.incremental_value.status == "INCREMENTAL_VALUE_CONFIRMED")
            n_confirmed_l2_permutation = sum(1 for results in research.level2_results.values() for r in results
                                              if r.test_result.status == "CONFIRMED" and r.permutation_test
                                              and r.permutation_test.status == "SURVIVES_PERMUTATION")
            n_confirmed_l2_stable = sum(1 for results in research.level2_results.values() for r in results
                                         if r.test_result.status == "CONFIRMED" and r.temporal_stability
                                         and r.temporal_stability.status == "STABLE_ACROSS_TIME")
            n_confirmed_l2_all_four = sum(1 for results in research.level2_results.values() for r in results
                                           if r.test_result.status == "CONFIRMED"
                                           and r.incremental_value and r.incremental_value.status == "INCREMENTAL_VALUE_CONFIRMED"
                                           and r.permutation_test and r.permutation_test.status == "SURVIVES_PERMUTATION"
                                           and r.temporal_stability and r.temporal_stability.status == "STABLE_ACROSS_TIME")
            n_confirmed_l3 = sum(1 for results in research.level3_results.values()
                                  for r in results if r.test_result.status == "CONFIRMED")

            if n_confirmed_l1 == 0 and n_confirmed_l2 == 0:
                print(f"\n  direction={direction} horizon={horizon}D: Level1 confirmed=0/{research.families_screened} "
                      "- nothing to report.")
                continue

            print(f"\n  direction={direction} horizon={horizon}D: "
                  f"Level1 confirmed={n_confirmed_l1}/{research.families_screened}  "
                  f"Level2 confirmed={n_confirmed_l2}  Level3 confirmed={n_confirmed_l3}")
            print(f"    Of the {n_confirmed_l2} Level-2 setups confirmed vs. baseline scalar: "
                  f"{n_confirmed_l2_incremental} also incremental, {n_confirmed_l2_permutation} also survive "
                  f"permutation, {n_confirmed_l2_stable} also temporally stable, "
                  f"{n_confirmed_l2_all_four} pass ALL FOUR checks.")
            for dimension, results in research.level2_results.items():
                _print_confirmed_detail(dimension, "L2", results)
            for setup_key, results in research.level3_results.items():
                _print_confirmed_detail(setup_key, "L3", results)

    n_technical_active = conn.execute(
        "SELECT COUNT(*) c FROM validated_relationships WHERE concept IS NOT NULL AND status = 'ACTIVE'"
    ).fetchone()["c"]
    n_technical_shadow = conn.execute(
        "SELECT COUNT(*) c FROM validated_relationships WHERE concept IS NOT NULL AND status = 'SHADOW'"
    ).fetchone()["c"]
    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"  Technical-concept relationships now ACTIVE: {n_technical_active}")
    print(f"  Technical-concept relationships now SHADOW: {n_technical_shadow}")
    print("  (Stage 6's flat approach confirmed ZERO out of 77,190 hypotheses tested. The counts above pass "
          "ONLY the existing baseline-scalar significance test - see the 'pass ALL FOUR checks' lines above "
          "for how many ALSO clear incremental-value, permutation, and temporal-stability diagnostics.)")


if __name__ == "__main__":
    main()
