"""CLI for the prospective evaluation harness.

    python -m market_agent.research.evaluation log NVDA AAPL MSFT
    python -m market_agent.research.evaluation resolve
    python -m market_agent.research.evaluation report --horizon 5

`log` triggers one real, live prediction pass (all three modes) per
entity given, at real current wall-clock time - see run.py's own
docstring for why this can never be pointed at a past date. `resolve`
runs outcome_resolution.py once over every entity ever logged. `report`
prints mode_report()'s comparison for one horizon - honestly empty/None
metrics if nothing has matured yet, never fabricated.
"""
from __future__ import annotations

import sys

from market_agent.research.evaluation.metrics_report import compare_modes
from market_agent.research.evaluation.outcome_resolution import resolve_outcomes
from market_agent.research.evaluation.run import log_predictions_for_watchlist
from market_agent.research.pipeline import DEFAULT_DB_PATH
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1

    conn = db.connect(DEFAULT_DB_PATH)
    command = argv[0]

    if command == "log":
        entities = [t.upper() for t in argv[1:]]
        if not entities:
            print("Usage: python -m market_agent.research.evaluation log TICKER [TICKER ...]")
            return 1
        results = log_predictions_for_watchlist(conn, entities)
        for entity, ids in results.items():
            print(f"{entity}: {ids}")
        return 0

    if command == "resolve":
        resolved = resolve_outcomes(conn, YahooPriceSeriesProvider())
        if not resolved:
            print("No predictions had a horizon mature since the last run - nothing to resolve.")
        for r in resolved:
            print(f"{r['entity']} [{r['mode']}] {r['horizon_trading_days']}d: "
                 f"abnormal_return={r['abnormal_return']:+.2%}")
        return 0

    if command == "report":
        horizon = 5
        if "--horizon" in argv:
            horizon = int(argv[argv.index("--horizon") + 1])
        comparison = compare_modes(conn, horizon)
        for mode, (metrics, portfolio) in comparison.items():
            print(f"--- {mode} (horizon={horizon}d) ---")
            print(f"  n={metrics.n} (excluded_no_prediction={metrics.n_excluded_no_prediction})")
            if metrics.n == 0:
                print("  No resolved observations yet at this horizon - no predictive-validity claim made.")
                continue
            print(f"  direction_accuracy={metrics.direction_accuracy}  brier={metrics.brier_score}")
            print(f"  sharpe={portfolio.sharpe_per_trade}  sortino={portfolio.sortino_per_trade}  "
                 f"max_drawdown={portfolio.max_drawdown}")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
