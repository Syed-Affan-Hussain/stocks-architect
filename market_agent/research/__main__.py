"""Item 21/23: the CLI.

    python -m market_agent.research --entity NVDA
    python -m market_agent.research --entity NVDA --json
    python -m market_agent.research watch NVDA
    python -m market_agent.research unwatch NVDA
    python -m market_agent.research list
    python -m market_agent.research watchlist

Uses ONLY the public API (pipeline.py/watchlist.py) - no pipeline logic
is duplicated here (item 22's explicit requirement).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from market_agent.research.pipeline import DEFAULT_DB_PATH, research_company
from market_agent.research.report_format import render_text
from market_agent.research.watchlist import list_watchlist, research_watchlist, unwatch, watch
from market_agent.store import db


def _print_report(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_text(report))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "watch" and len(argv) > 1:
        conn = db.connect(DEFAULT_DB_PATH)
        watch(conn, argv[1])
        print(f"Watching {argv[1].upper()}.")
        return 0
    if argv and argv[0] == "unwatch" and len(argv) > 1:
        conn = db.connect(DEFAULT_DB_PATH)
        unwatch(conn, argv[1])
        print(f"Stopped watching {argv[1].upper()}.")
        return 0
    if argv and argv[0] == "list":
        conn = db.connect(DEFAULT_DB_PATH)
        watched = list_watchlist(conn)
        print("\n".join(watched) if watched else "Watchlist is empty.")
        return 0
    if argv and argv[0] == "watchlist":
        conn = db.connect(DEFAULT_DB_PATH)
        watched = list_watchlist(conn)
        if not watched:
            print("Watchlist is empty. Use: python -m market_agent.research watch <TICKER>")
            return 0
        as_json = "--json" in argv
        for report in research_watchlist(conn):
            _print_report(report, as_json)
        return 0

    parser = argparse.ArgumentParser(prog="market_agent.research",
                                      description="AI Market Research & Analysis - research a public company.")
    parser.add_argument("--entity", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of the text report.")
    parser.add_argument("--lookback-days", type=int, default=30, help="News/event lookback window in days.")
    args = parser.parse_args(argv)

    conn = db.connect(DEFAULT_DB_PATH)
    report = research_company(args.entity, conn=conn, lookback_days=args.lookback_days)
    _print_report(report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
