"""One full autonomous cycle of the prospective evaluation harness - what
.github/workflows/prospective_eval.yml runs on every scheduled trigger,
and what a manual cron/Task-Scheduler entry on a local machine could run
too. Reads the watchlist, logs a fresh prediction pass for every entity in
it, resolves any outcome that has genuinely matured since last time, then
regenerates the static dashboard.

    python scripts/run_prospective_cycle.py

Reads the watchlist from watchlist.txt at the repo root (one ticker per
line, '#' comments allowed, blank lines ignored) - a plain text file
specifically so it can be edited directly in GitHub's web UI without
needing to touch the SQLite database at all. This is DELIBERATELY
SEPARATE from the DB-backed research_watchlist table watchlist.py already
provides for local CLI use (item 23) - that mechanism still works
unchanged for anyone running this locally; this script's watchlist.txt is
the one the GitHub Actions workflow reads, because a binary SQLite file
isn't something you can sanely hand-edit on github.com.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_agent.research.evaluation.outcome_resolution import resolve_outcomes
from market_agent.research.evaluation.run import log_predictions_for_watchlist
from market_agent.research.evaluation.static_dashboard import generate
from market_agent.research.pipeline import DEFAULT_DB_PATH
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

WATCHLIST_PATH = "watchlist.txt"
DASHBOARD_OUT = "docs/index.html"


def read_watchlist(path: str = WATCHLIST_PATH) -> list[str]:
    file = Path(path)
    if not file.exists():
        return []
    tickers = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line:
            tickers.append(line)
    return tickers


def main() -> int:
    tickers = read_watchlist()
    if not tickers:
        print(f"{WATCHLIST_PATH} is empty or missing - nothing to log this cycle. "
              f"Add tickers (one per line) to start tracking.")
    conn = db.connect(DEFAULT_DB_PATH)
    prices = YahooPriceSeriesProvider()

    if tickers:
        print(f"Logging predictions for: {', '.join(tickers)}")
        results = log_predictions_for_watchlist(conn, tickers, prices=prices)
        for entity, ids in results.items():
            print(f"  {entity}: {ids}")

    print("Resolving matured outcomes...")
    resolved = resolve_outcomes(conn, prices)
    if not resolved:
        print("  Nothing matured since the last run.")
    for r in resolved:
        print(f"  {r['entity']} [{r['mode']}] {r['horizon_trading_days']}d: abnormal_return={r['abnormal_return']:+.2%}")

    conn.close()

    print(f"Regenerating {DASHBOARD_OUT}...")
    # GITHUB_REPOSITORY ("owner/repo") is set automatically by GitHub Actions on every run - never
    # guessed, and correctly None when this script is run locally outside Actions.
    generate(DEFAULT_DB_PATH, DASHBOARD_OUT, repo=os.environ.get("GITHUB_REPOSITORY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
