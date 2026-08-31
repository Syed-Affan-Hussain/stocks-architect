"""Item 23: a simple persistent watchlist over the SAME research_reports/
research_watchlist ledger (store/schema.py v5)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from market_agent.research.pipeline import research_company
from market_agent.research.schema import ResearchReport
from market_agent.store import db


def watch(conn: sqlite3.Connection, ticker: str) -> None:
    db.add_to_watchlist(conn, ticker.upper().strip(), datetime.now(timezone.utc))


def unwatch(conn: sqlite3.Connection, ticker: str) -> None:
    db.remove_from_watchlist(conn, ticker.upper().strip())


def list_watchlist(conn: sqlite3.Connection) -> list[str]:
    return db.get_watchlist(conn)


def research_watchlist(conn: sqlite3.Connection, **kwargs) -> list[ResearchReport]:
    """Item 23: `research watchlist` - runs research_company for every
    watched entity and returns the reports in watchlist order."""
    return [research_company(ticker, conn=conn, **kwargs) for ticker in list_watchlist(conn)]
