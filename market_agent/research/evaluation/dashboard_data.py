"""Shared data-collection for both dashboard surfaces (dashboard.py's live
local server and static_dashboard.py's GitHub Pages generator) - one
function, so the two never drift into showing different numbers for the
same underlying database.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from market_agent.outcomes.ohlcv import OHLCVProvider
from market_agent.research.evaluation.metrics_report import mode_report
from market_agent.research.evaluation.modes import MODES
from market_agent.research.evaluation.outcome_resolution import HORIZONS
from market_agent.research.pipeline import DEFAULT_DB_PATH
from market_agent.store import db

IMPLICATION_AXES = ("growth", "profitability", "cash_flow", "balance_sheet", "demand", "supply_chain",
                     "competitive_position", "regulatory", "guidance", "risk")
PRICE_LOOKBACK_DAYS = 180


def _collect_price_series(ohlcv: OHLCVProvider, entities: list[str], as_of: datetime,
                           lookback_days: int = PRICE_LOOKBACK_DAYS) -> dict[str, list[dict]]:
    """Real historical OHLCV bars per entity, in TradingView lightweight-
    charts' own {time, open, high, low, close} shape - a genuine spot-price
    history, not a placeholder. Entities with no retrievable price history
    are simply absent from the result (never a fabricated flat line)."""
    series: dict[str, list[dict]] = {}
    for entity in entities:
        try:
            bars = ohlcv.bars(entity, as_of, lookback_days)
        except Exception:  # noqa: BLE001 - one entity's price-fetch failure must not break the whole dashboard
            bars = []
        # A same-day bar fetched while the market is still open (or just after a data-provider gap)
        # can come back with NaN OHLC values - not a real price, and `NaN` embedded directly into the
        # page's JS would silently break the candlestick series. Dropped, never coerced to 0/null-as-
        # zero, which would plot a fake price instead of correctly having one fewer, real bar.
        clean = [b for b in bars if not any(math.isnan(v) for v in (b.open, b.high, b.low, b.close))]
        if clean:
            series[entity] = [{"time": b.date.date().isoformat(), "open": b.open, "high": b.high,
                                "low": b.low, "close": b.close} for b in clean]
    return series


def collect_dashboard_data(conn: sqlite3.Connection, ohlcv: OHLCVProvider | None = None,
                            as_of: datetime | None = None, repo: str | None = None,
                            branch: str = "master") -> dict:
    """`repo`: "owner/name" if known (static_dashboard.py passes this from
    --repo, sourced from GitHub Actions' own `github.repository` context
    var - never guessed). Used only to build `repo_edit_url`, the deep
    link dashboard_template.py's static "Add via GitHub" control opens -
    None if not supplied, which the template treats as "no repo link
    configured" rather than fabricating one from a guessed path."""
    as_of = as_of or datetime.now(timezone.utc)
    rows = db.all_predictions(conn)
    predictions = []
    for r in rows:
        snapshot = json.loads(r["inputs_snapshot_json"])
        predictions.append({
            "id": r["prediction_id"], "entity": r["entity"], "mode": r["mode"],
            "decision_label": r["decision_label"], "predicted_impact": r["predicted_impact"],
            "predicted_confidence": r["predicted_confidence"], "triggered_at": r["triggered_at"],
            "model_version": r["model_version"],
            "realized_return_1d": r["realized_return_1d"], "realized_return_5d": r["realized_return_5d"],
            "realized_return_20d": r["realized_return_20d"], "realized_return_60d": r["realized_return_60d"],
            "assessment_confidence": snapshot.get("assessment_confidence"),
            "narrative_count": snapshot.get("narrative_count"), "risk_count": snapshot.get("risk_count"),
            "llm_status": snapshot.get("llm_status"), "mode_reasoning": snapshot.get("mode_reasoning"),
            "news_state": snapshot.get("news_state"),
        })

    metrics: dict[str, dict[str, dict]] = {}
    for trading_days, _, _, _ in HORIZONS:
        by_mode = {}
        for mode in MODES:
            m, p = mode_report(conn, mode, trading_days)
            by_mode[mode] = {
                "n": m.n, "direction_accuracy": m.direction_accuracy, "mae": m.mae, "rmse": m.rmse,
                "brier_score": m.brier_score, "spearman_r": m.spearman_r,
                "sharpe_per_trade": p.sharpe_per_trade, "sharpe_annualized": p.sharpe_annualized,
                "sortino_per_trade": p.sortino_per_trade, "sortino_annualized": p.sortino_annualized,
                "max_drawdown": p.max_drawdown, "turnover_trades_per_year": p.turnover_trades_per_year,
                "expected_value_after_costs": p.expected_value_after_costs,
            }
        metrics[str(trading_days)] = by_mode

    entities = sorted({r["entity"] for r in rows})
    price_series = _collect_price_series(ohlcv, entities, as_of) if ohlcv is not None else {}
    repo_edit_url = f"https://github.com/{repo}/edit/{branch}/watchlist.txt" if repo else None

    return {"predictions": predictions, "metrics": metrics, "modes": list(MODES), "axes": list(IMPLICATION_AXES),
            "db_path": DEFAULT_DB_PATH, "price_series": price_series, "generated_at": as_of.isoformat(),
            "repo_edit_url": repo_edit_url}
