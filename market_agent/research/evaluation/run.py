"""Orchestrates ONE prospective evaluation pass: run the existing research
pipeline exactly once per entity (real network calls - SEC/Yahoo/Google
News, all live, all "as of now" - see pipeline.py's own disclosed
point-in-time limitation), compute all three modes from that SINGLE
report, and log each as its own immutable prediction_log row.

Deliberately calls research_company() with NO `generated_at` override -
every prospective prediction is triggered at real, current wall-clock
time. There is no code path here that can be pointed at a past date to
retroactively fabricate a prediction as if it had been made then; see
pipeline.py's module docstring for why that would be unsound anyway
(providers fetch "live now", not a genuine historical snapshot).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from market_agent.llm.interpreter import LLMClient
from market_agent.research.evaluation.decision_mapping import MODEL_VERSION
from market_agent.research.evaluation.modes import compute_all_modes
from market_agent.research.pipeline import DEFAULT_DB_PATH, research_company
from market_agent.research.schema import ResearchReport
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db


def _inputs_snapshot(report: ResearchReport) -> dict:
    """A disclosed, auditable record of exactly what fed this prediction -
    not the full report (that's already persisted separately in
    research_reports), just enough to answer "why did the model decide
    this" without re-running the pipeline."""
    return {
        "entity": report.entity, "generated_at": report.generated_at, "llm_status": report.llm_status,
        "assessment": report.assessment, "assessment_confidence": report.assessment_confidence,
        "narrative_count": len(report.narratives), "risk_count": len(report.risks),
        "consistency_check_count": len(report.consistency_checks),
        "news_state": report.news_state, "unavailable_sources": report.unavailable_sources,
    }


def log_predictions_for_entity(entity: str, conn: sqlite3.Connection,
                                prices: YahooPriceSeriesProvider | None = None,
                                llm_client: LLMClient | None = None) -> list[str]:
    """Runs the pipeline once, logs all three modes, returns the three new
    prediction_id's. Raises nothing special on missing data - a
    SOURCE_UNAVAILABLE news/fundamentals pass still produces a valid
    report (see pipeline.py), which still yields three (likely low- or
    no-signal) logged predictions rather than silently skipping the
    entity."""
    triggered_at = datetime.now(timezone.utc)
    report = research_company(entity, conn=conn, prices=prices, llm_client=llm_client, generated_at=triggered_at)
    snapshot = _inputs_snapshot(report)

    prediction_ids = []
    for result in compute_all_modes(report):
        prediction_ids.append(db.save_prediction(
            conn, entity=entity, mode=result.mode, triggered_at=triggered_at, model_version=MODEL_VERSION,
            decision_label=result.decision.decision_label, predicted_impact=result.decision.predicted_impact,
            predicted_confidence=result.decision.predicted_confidence,
            inputs_snapshot={**snapshot, "mode_reasoning": result.reasoning},
        ))
    return prediction_ids


def log_predictions_for_watchlist(conn: sqlite3.Connection, entities: list[str],
                                   prices: YahooPriceSeriesProvider | None = None) -> dict[str, list[str]]:
    prices = prices or YahooPriceSeriesProvider()
    results: dict[str, list[str]] = {}
    for entity in entities:
        try:
            results[entity] = log_predictions_for_entity(entity, conn, prices=prices)
        except Exception as e:  # noqa: BLE001 - one entity's failure must not abort the whole batch
            results[entity] = [f"FAILED: {e!r}"]
    return results
