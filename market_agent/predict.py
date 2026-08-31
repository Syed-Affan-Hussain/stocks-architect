"""Real-time single-event prediction, logged to the immutable prediction
ledger - stage 5, items 1/2.

`run_predict()` is the network-free-testable core (callers inject `conn`
and `prices`); `main()` is the thin argparse wrapper that wires up the
real YahooPriceSeriesProvider and a persistent sqlite ledger, invoked via:

    python -m market_agent.predict --entity NVDA --text "NVDA raises full-year guidance"

This does not invent a new prediction mechanism - it is pipeline.py's
already-tested interpret -> context -> predict assembly, plus the one
piece pipeline.py explicitly leaves to its caller (item 1's docstring):
actually writing the result to episodic_events via store.db.log_prediction,
so a live call becomes a permanent, auditable ledger row exactly like a
walk-forward-harness prediction does.

BASELINE SOURCE: the unconditional baseline is estimated from every
already-resolved episodic_events row of this event_type in the ledger
itself (estimate_baseline_from_ledger, below) - not a fresh burn-in split
recomputed on every invocation. A continuously operating system's baseline
should improve as real outcomes accumulate in the shared ledger; falls
back to the same disclosed 0.02 default experiment/walkforward.py's
_estimate_baseline uses when no resolved history exists yet (e.g. the very
first call against a brand-new ledger).

ABSTAINING IS A VALID, EXPECTED OUTPUT: this may report NOT_RECOGNIZED
(the interpreter found no in-scope event in the text) or
INSUFFICIENT_PRECEDENT per horizon (see pipeline.py) - both are printed
plainly, never silently upgraded to a fabricated number.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

from market_agent.agents.adaptive_agent import AdaptiveAgent
from market_agent.events.schema import PredictionRecord
from market_agent.llm.select import select_interpreter_from_env
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.pipeline import predict_event
from market_agent.store import db

DEFAULT_LEDGER_PATH = "data_cache/prediction_ledger.sqlite"
DEFAULT_HORIZONS = [1, 5, 20, 60]
DEFAULT_BENCHMARK = "SPY"
FALLBACK_BASELINE_MAGNITUDE = 0.02  # same disclosed default as experiment/walkforward.py::_estimate_baseline


def estimate_baseline_from_ledger(conn: sqlite3.Connection, event_type: str,
                                   horizon_days_list: list[int]) -> dict[int, float]:
    baseline = {}
    for horizon_days in horizon_days_list:
        rows = conn.execute(
            """SELECT realized_abnormal_return FROM episodic_events
               WHERE event_type = ? AND horizon_days = ? AND outcome_locked = 1
                 AND realized_abnormal_return IS NOT NULL""",
            (event_type, horizon_days)).fetchall()
        magnitudes = [abs(r["realized_abnormal_return"]) for r in rows]
        baseline[horizon_days] = (sum(magnitudes) / len(magnitudes)) if magnitudes else FALLBACK_BASELINE_MAGNITUDE
    return baseline


def run_predict(conn: sqlite3.Connection, prices: PriceSeriesProvider, entity: str, raw_text: str, source: str,
                 published_at: datetime, predicted_at: datetime | None = None, interpreter=None,
                 horizon_days_list: list[int] | None = None) -> dict:
    """The CLI's live entrypoint: pipeline.predict_event() (the reusable,
    non-logging composition) plus the one thing that's this caller's own
    decision - actually writing each horizon's prediction to the
    persistent, immutable ledger via store.db.log_prediction, so a live
    call becomes a permanent, auditable row exactly like a walk-forward
    prediction does. Returns {"status": "NOT_RECOGNIZED", "entity",
    "raw_text"} or {"status": "OK", "event", "predictions":
    list[SecurityPrediction], "logged_event_ids", "interpreter_name",
    "agent_model_version"}."""
    predicted_at = predicted_at or published_at
    horizon_days_list = horizon_days_list or DEFAULT_HORIZONS
    interpreter = interpreter or select_interpreter_from_env()

    # event_type isn't known until interpretation happens inside predict_event() - but the baseline
    # and agent need to exist before it can run. Every event type this system currently understands
    # is GUIDANCE_CHANGE (events/interpret.py), so that's what the baseline is estimated against;
    # this line is the one place that assumption would need revisiting if a second event type's
    # interpreter were wired into select_interpreter_from_env() without updating this CLI too.
    baseline = estimate_baseline_from_ledger(conn, "GUIDANCE_CHANGE", horizon_days_list)
    agent = AdaptiveAgent(conn, baseline)

    result = predict_event(conn, agent, prices, interpreter, entity, raw_text, source, published_at,
                            predicted_at, horizon_days_list, DEFAULT_BENCHMARK)
    if result.status == "NOT_RECOGNIZED":
        return {"status": "NOT_RECOGNIZED", "entity": entity, "raw_text": raw_text}

    knowledge_version = db.count_governance_changes(conn)
    logged_ids = []
    for sp in result.predictions:
        record = PredictionRecord(horizon_days=sp.horizon_days, predicted_impact=sp.predicted_impact,
                                   predicted_confidence=sp.predicted_confidence, basis=sp.basis,
                                   model_version=agent.model_version, predicted_at=predicted_at,
                                   uncertainty=sp.uncertainty, retrieved_cases=sp.similar_case_ids,
                                   knowledge_version=knowledge_version, novelty_score=sp.novelty_score)
        logged_ids.append(db.log_prediction(conn, result.event, record))

    return {"status": "OK", "event": result.event, "predictions": result.predictions, "logged_event_ids": logged_ids,
            "interpreter_name": interpreter.NAME, "agent_model_version": agent.model_version}


def format_report(result: dict) -> str:
    if result["status"] == "NOT_RECOGNIZED":
        return (f"SECURITY: {result['entity']}\n"
                f"EVENT: NOT_RECOGNIZED - no in-scope event pattern matched this text.\n"
                f"  text: {result['raw_text']!r}\n")

    event = result["event"]
    lines = [
        f"SECURITY: {event.entity}",
        f"EVENT: {event.event_type}",
        f"DIRECTION: {event.direction}",
        f"REGIME: {event.context.get('regime', 'UNKNOWN')}",
    ]
    for sp in result["predictions"]:
        novelty = f"{sp.novelty_score:.2f}"
        precedent = f"{sp.n_similar_cases} similar case(s)"
        if sp.status == "INSUFFICIENT_PRECEDENT":
            lines.append(f"  {sp.horizon_days}D: INSUFFICIENT_PRECEDENT (novelty={novelty}, {precedent})")
        else:
            impact = f"{sp.predicted_impact:+.2%}" if sp.predicted_impact is not None else "n/a"
            uncertainty = f" +/-{sp.uncertainty:.2%}" if sp.uncertainty is not None else ""
            lines.append(f"  {sp.horizon_days}D: predicted {impact}{uncertainty} "
                         f"(confidence={sp.predicted_confidence}, novelty={novelty}, {precedent})")
    lines.append("EVIDENCE:")
    for sp in result["predictions"]:
        lines.append(f"  [{sp.horizon_days}D]")
        for r in sp.reasoning_provenance:
            lines.append(f"    - {r}")
    lines.append(f"(interpreter={result['interpreter_name']}, agent={result['agent_model_version']}, "
                 f"logged {len(result['logged_event_ids'])} ledger row(s) - see store/schema.py's episodic_events)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m market_agent.predict",
                                      description="Real-time single-event prediction, logged to the immutable "
                                                   "prediction ledger.")
    parser.add_argument("--entity", required=True, help="ticker, e.g. NVDA")
    parser.add_argument("--text", required=True, help="raw item text, e.g. a filing/press-release headline")
    parser.add_argument("--source", default="cli", help="source label for this raw item")
    parser.add_argument("--published-at", default=None,
                         help="ISO timestamp the item was published (default: now, UTC)")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                         help="comma-separated horizon days, e.g. 1,5,20,60")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, help="sqlite path for the persistent ledger")
    args = parser.parse_args(argv)

    published_at = (datetime.fromisoformat(args.published_at) if args.published_at
                     else datetime.now(timezone.utc))
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    horizon_days_list = [int(h) for h in args.horizons.split(",")]

    from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider  # local: avoids a network-capable
    #                                                                          import at module load time for
    #                                                                          test callers of run_predict().
    conn = db.connect(args.ledger)
    prices = YahooPriceSeriesProvider()
    result = run_predict(conn, prices, args.entity.upper(), args.text, args.source, published_at,
                          horizon_days_list=horizon_days_list)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
