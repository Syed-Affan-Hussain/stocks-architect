"""Generates a fully static HTML dashboard - no backend, no live /api
endpoints - for GitHub Pages. Same page shell and data shape as the live
local dashboard.py (both share dashboard_data.py/dashboard_template.py),
so the two never look or behave differently for the same underlying data;
the only real difference is the "Track a ticker" control, which has no
live backend to call here (see dashboard_template.py's static branch).

    python -m market_agent.research.evaluation.static_dashboard \
        --db data_cache/research/market_agent_research.sqlite \
        --out docs/index.html

This is what .github/workflows/prospective_eval.yml runs after log/resolve
on every scheduled run, so the committed docs/index.html - and therefore
the GitHub Pages site built from it - reflects the real, current
prediction log and real, current price history each time, not a stale
one-off snapshot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from market_agent.research.evaluation.dashboard_data import collect_dashboard_data
from market_agent.research.evaluation.dashboard_template import render_page
from market_agent.research.pipeline import DEFAULT_DB_PATH
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db


def generate(db_path: str, out_path: str) -> None:
    conn = db.connect(db_path)
    try:
        data = collect_dashboard_data(conn, ohlcv=YahooPriceSeriesProvider())
    finally:
        conn.close()
    html = render_page(data, static=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html):,} bytes) - {len(data['predictions'])} predictions, "
          f"{len(data['price_series'])} entities with price history.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args()
    generate(args.db, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
