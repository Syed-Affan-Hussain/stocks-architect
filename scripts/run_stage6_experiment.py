"""Stage 6 real-data run: methodology ingestion + technical concepts +
the four-way walk-forward comparison (STATIC / CURRENT ADAPTIVE /
TECHNICAL ADAPTIVE / METHODOLOGY-INFORMED ADAPTIVE), evaluated at 1D/5D/
20D/60D with direction accuracy, MAE, RMSE, Brier/calibration, Sharpe, max
drawdown, turnover, transaction-cost-adjusted return, and expected value -
on real SEC EDGAR events and real Yahoo Finance OHLCV.

    python scripts/run_stage6_experiment.py

Reuses stage 5's cached EDGAR event data (data_cache/edgar/*.json) rather
than re-fetching it - nothing about stage 6 changes how those events are
sourced. Price data IS re-fetched per ticker on first read here, because
outcomes/ohlcv.py's OHLCV cache format (open/high/low/close/volume) is a
superset of stage 4-5's close-only cache - see sources/yahoo_prices.py's
module docstring for the transparent, disclosed migration.

CONFIGURATION FIXED BEFORE THIS RUN: every constant this run depends on
(horizons, embargo, burn-in/holdout fractions, MIN_N/ALPHA in
hypothesis_testing.py, shadow-probation size, technical-concept
thresholds in concepts/technical_context.py, MAX_CONDITIONING_VARS and
MAX_TECHNICAL_DIMENSIONS_PER_EVENT in learn/hypothesis.py,
TRANSACTION_COST_PER_TRADE in experiment/portfolio_metrics.py) was fixed
before this run and is not retuned based on what it reports - same
standing rule as every prior real-data run in this project.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.experiment.four_way_walkforward import (
    AGENT_NAMES, FourWayWalkforwardConfig, run_four_way_walkforward,
)
from market_agent.experiment.metrics import PredictionOutcome, compute_metrics
from market_agent.experiment.portfolio_metrics import TradeRecord, compute_portfolio_metrics
from market_agent.learn.hypothesis import RuleBasedHypothesisGenerator
from market_agent.methodology.extractor import RuleBasedMethodologyExtractor
from market_agent.methodology.ingest import ingest_corpus
from market_agent.methodology.seed_corpus import SEED_CORPUS
from market_agent.pit.clock import PointInTimeClock
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
LEDGER_PATH = "data_cache/stage6_experiment.sqlite"
HORIZONS = [1, 5, 20, 60]


def _print_metrics_block(title: str, outcomes: list[PredictionOutcome]) -> None:
    m = compute_metrics(outcomes)
    print(f"\n  {title} (n={m.n}, {m.n_excluded_no_prediction} excluded)")
    if m.n < 2:
        print("    Too few scoreable predictions for metrics.")
        return
    acc = f"{m.direction_accuracy:.1%}" if m.direction_accuracy is not None else "n/a"
    print(f"    Direction accuracy: {acc}   MAE: {m.mae:.2%}   RMSE: {m.rmse:.2%}   "
          f"Brier: {m.brier_score:.3f}")


def _print_portfolio_block(title: str, trades: list[TradeRecord]) -> None:
    p = compute_portfolio_metrics(trades)
    print(f"\n  {title} (n_trades={p.n_trades})")
    if p.n_trades < 2:
        print(f"    {p.evidence[0]}")
        return
    ev = f"{p.expected_value_after_costs:+.4f}"
    sharpe = f"{p.sharpe_annualized:.2f}" if p.sharpe_annualized is not None else "n/a"
    print(f"    EV(after costs): {ev}   Total return(after costs): {p.total_return_after_costs:+.4f}   "
          f"Sharpe(annualized): {sharpe}   Max drawdown: {p.max_drawdown:.4f}   "
          f"Turnover: {p.turnover_trades_per_year:.1f} trades/yr")


def _trades_for(scored, agent: str, horizon: int, segment: str) -> list[TradeRecord]:
    return [TradeRecord(entity=s.entity, triggered_at=s.published_at.isoformat(), horizon_days=horizon,
                         predicted_impact=s.predicted_impact, realized_abnormal_return=s.realized_abnormal_return)
            for s in scored
            if s.agent == agent and s.horizon_days == horizon and s.segment == segment
            and s.predicted_impact is not None and s.realized_abnormal_return is not None]


def main() -> None:
    clock = PointInTimeClock(now=datetime.now(timezone.utc))
    Path("data_cache").mkdir(exist_ok=True)
    ledger_path = Path(LEDGER_PATH)
    if ledger_path.exists():
        ledger_path.unlink()  # fresh run - this script's own scratch ledger
    conn = db.connect(str(ledger_path))

    print("[1/6] Ingesting the methodology seed corpus...")
    methodology_ids = ingest_corpus(conn, RuleBasedMethodologyExtractor(), SEED_CORPUS, clock.now)
    print(f"  {len(methodology_ids)} methodologies ingested (see methodology/seed_corpus.py's disclosed "
          "small-proof-of-concept scope).")

    print(f"\n[2/6] Loading cached real SEC EDGAR events, {START_DATE} to {END_DATE}...")
    guidance_sourced = fetch_guidance_events(START_DATE, END_DATE, clock, cache_path=GUIDANCE_CACHE,
                                              max_pages_per_phrase=10)
    dividend_sourced = fetch_dividend_events(START_DATE, END_DATE, clock, cache_path=DIVIDEND_CACHE,
                                              max_pages_per_phrase=10)
    print(f"  GUIDANCE_CHANGE: {len(guidance_sourced)} events.  DIVIDEND_CHANGE: {len(dividend_sourced)} events.")

    entities = sorted({s.raw_item.entity for s in guidance_sourced} | {s.raw_item.entity for s in dividend_sourced})
    print(f"\n[3/6] Loading real Yahoo Finance OHLCV for {len(entities)} entities + SPY "
          "(re-fetches into the new OHLCV cache format - see module docstring)...")
    prices = YahooPriceSeriesProvider(cache_dir="data_cache/prices")
    prices._load_frame("SPY")
    n_ok = 0
    for ticker in entities:
        try:
            n_ok += 0 if prices._load_frame(ticker).empty else 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  {n_ok}/{len(entities)} entities have usable OHLCV history.")

    print("\n[4/6] Running the four-way walk-forward comparison for both event types...")
    config = FourWayWalkforwardConfig(horizon_days_list=HORIZONS, embargo_days=2, benchmark_ticker="SPY",
                                       burn_in_fraction=0.20, final_holdout_fraction=0.20)
    guidance_report = run_four_way_walkforward(guidance_sourced, prices, RuleBasedInterpreter(),
                                                RuleBasedHypothesisGenerator(), config, conn, ohlcv=prices)
    dividend_report = run_four_way_walkforward(dividend_sourced, prices, RuleBasedInterpreter(),
                                                RuleBasedHypothesisGenerator(), config, conn, ohlcv=prices)

    print("\n[5/6] Scoring on the FINAL_HOLDOUT segment only (untouched, evaluated once)...")
    for label, report in (("GUIDANCE_CHANGE", guidance_report), ("DIVIDEND_CHANGE", dividend_report)):
        print(f"\n{'=' * 78}\n{label} - FINAL HOLDOUT RESULTS\n{'=' * 78}")
        for line in report.evidence:
            print(f"  {line}")
        for horizon in HORIZONS:
            print(f"\n  --- {label} horizon {horizon}D (FINAL_HOLDOUT) ---")
            for agent in AGENT_NAMES:
                rows = [s for s in report.scored if s.agent == agent and s.horizon_days == horizon
                        and s.segment == "FINAL_HOLDOUT"]
                outcomes = [PredictionOutcome(s.predicted_impact, s.predicted_confidence, s.realized_abnormal_return)
                            for s in rows if s.realized_abnormal_return is not None]
                _print_metrics_block(agent, outcomes)
                trades = _trades_for(report.scored, agent, horizon, "FINAL_HOLDOUT")
                _print_portfolio_block(f"{agent} (economic)", trades)

    print("\n[6/6] Building the concept-level / methodology-level knowledge-state report...")
    knowledge = build_knowledge_state_report(conn, now=datetime.now(timezone.utc))
    print(f"\n{'=' * 78}\nKNOWLEDGE-STATE DASHBOARD\n{'=' * 78}")
    print(format_knowledge_report(knowledge))

    print(f"\n{'=' * 78}\nSTAGE 6 SUCCESS CRITERION\n{'=' * 78}")
    print("  Success is: the four-way comparison ran end to end on real data with technical concepts and "
          "methodology provenance both live, no statistical gate was loosened, no threshold was retuned "
          "after seeing these numbers, and the final holdout was touched exactly once. Whether TECHNICAL "
          "or METHODOLOGY-INFORMED ADAPTIVE outperformed CURRENT ADAPTIVE or STATIC on this holdout is "
          "reported above as obtained - a negative or mixed result is not a failure of this criterion.")


if __name__ == "__main__":
    main()
