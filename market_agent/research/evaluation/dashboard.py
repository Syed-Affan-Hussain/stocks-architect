"""A live, local "terminal"-style dashboard for the prospective evaluation
harness - stdlib only on the Python side (http.server), no new pip
dependency. Reads market_agent_research.sqlite FRESH on every request/API
call; nothing is cached or pre-computed, so the page always reflects
whatever `log`/`resolve`/the in-page "Track" control have actually
written. Real historical price data is fetched live via
YahooPriceSeriesProvider for the spot-price chart.

    python -m market_agent.research.evaluation.dashboard [--port 8765]

THREE ENDPOINTS:
  GET  /            full page shell with the CURRENT data embedded as JSON
                     (window.__DATA__) - all filtering/sorting/charting
                     happens client-side in vanilla JS, no framework. See
                     dashboard_template.py for the shared page shell (also
                     used by static_dashboard.py's GitHub Pages build) and
                     dashboard_data.py for the shared data collection.
  GET  /api/data     the same JSON, for the page's own "Refresh" action
                     without a full reload.
  POST /api/track    body {"ticker": "XXXX"} - runs ONE real, live
                     research_company() pass (real SEC/Yahoo/Google News
                     calls, several seconds) via run.log_predictions_for_
                     entity, exactly the same code path `python -m
                     market_agent.research.evaluation log` uses. Only
                     available here (the local live server) - the static
                     GitHub Pages build has no backend to run this; see
                     that build's watchlist.txt mechanism instead.

GRAPHS ARE REAL DATA OR AN HONEST EMPTY STATE, NEVER A PLACEHOLDER FAKE:
the price/impact/confidence/news-axis charts plot today's actual data.
The equity-curve and per-mode performance charts run through the exact
same compute_portfolio_metrics/compute_metrics path the CLI `report`
command uses - with zero resolved outcomes today, they render an explicit
"no resolved observations yet" state instead of an empty or fabricated
line, for the same reason metrics_report.py's own functions never
fabricate a value from n=0.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from market_agent.research.evaluation.dashboard_data import collect_dashboard_data
from market_agent.research.evaluation.dashboard_template import render_page
from market_agent.research.evaluation.run import log_predictions_for_entity
from market_agent.research.pipeline import DEFAULT_DB_PATH
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

DEFAULT_PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _collect(self) -> dict:
        conn = db.connect(DEFAULT_DB_PATH)
        try:
            return collect_dashboard_data(conn, ohlcv=YahooPriceSeriesProvider())
        finally:
            conn.close()

    def do_GET(self):  # noqa: N802
        if self.path in ("/", ""):
            body = render_page(self._collect(), static=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/data":
            self._send_json(self._collect())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path != "/api/track":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            ticker = str(body.get("ticker", "")).strip().upper()
            if not ticker:
                raise ValueError("No ticker supplied.")
            conn = db.connect(DEFAULT_DB_PATH)
            try:
                ids = log_predictions_for_entity(ticker, conn, prices=YahooPriceSeriesProvider())
            finally:
                conn.close()
            self._send_json({"ok": True, "ticker": ticker, "n_logged": len(ids)})
        except Exception as e:  # noqa: BLE001 - report the real failure to the page, don't crash the server
            self._send_json({"ok": False, "error": repr(e)}, status=500)

    def log_message(self, format, *args):  # noqa: A002
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard live at http://127.0.0.1:{args.port}/ (reads {DEFAULT_DB_PATH} fresh on every request)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
