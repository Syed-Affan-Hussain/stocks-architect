from datetime import datetime, timedelta, timezone

from market_agent.events.interpret import RuleBasedInterpreter
from market_agent.outcomes.observe import PriceSeriesProvider
from market_agent.predict import estimate_baseline_from_ledger, format_report, run_predict
from market_agent.store import db

PUBLISHED = datetime(2024, 6, 1, tzinfo=timezone.utc)


class FlatPrices(PriceSeriesProvider):
    """Every ticker flat at 100 (benchmark included) - deterministic,
    network-free: NORMAL regime, zero trailing returns, zero realized vol.
    Good enough to exercise the pipeline's control flow without asserting
    anything about the (necessarily trivial) numeric context values."""

    def close_price(self, ticker, as_of):
        return 100.0


def test_unrecognized_text_returns_not_recognized_without_logging():
    conn = db.connect(":memory:")
    prices = FlatPrices()
    result = run_predict(conn, prices, "NVDA", "NVDA corp announces a new product", "cli", PUBLISHED,
                          interpreter=RuleBasedInterpreter())
    assert result["status"] == "NOT_RECOGNIZED"
    assert conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"] == 0


def test_recognized_guidance_event_logs_one_row_per_horizon():
    conn = db.connect(":memory:")
    prices = FlatPrices()
    result = run_predict(conn, prices, "NVDA", "NVDA corp raises full-year guidance", "cli", PUBLISHED,
                          interpreter=RuleBasedInterpreter(), horizon_days_list=[1, 5, 20])
    assert result["status"] == "OK"
    assert result["event"].event_type == "GUIDANCE_CHANGE"
    assert result["event"].direction == "positive"
    assert len(result["predictions"]) == 3
    assert len(result["logged_event_ids"]) == 3
    assert conn.execute("SELECT COUNT(*) c FROM episodic_events").fetchone()["c"] == 3
    # every logged row is a real, immutable episodic_events row - queryable back out
    for event_id in result["logged_event_ids"]:
        assert db.get_event(conn, event_id) is not None


def test_estimate_baseline_from_ledger_falls_back_when_no_resolved_history():
    conn = db.connect(":memory:")
    baseline = estimate_baseline_from_ledger(conn, "GUIDANCE_CHANGE", [1, 5])
    assert baseline == {1: 0.02, 5: 0.02}


def test_estimate_baseline_from_ledger_averages_real_resolved_outcomes():
    from market_agent.events.schema import EventRecord, PredictionRecord
    conn = db.connect(":memory:")
    for i, magnitude in enumerate([0.10, 0.20]):
        published = PUBLISHED + timedelta(days=i)
        event = EventRecord(entity=f"E{i}", event_type="GUIDANCE_CHANGE", direction="positive", source="wire",
                             source_reliability_snapshot=0.5, raw_text="x", published_at=published,
                             ingested_at=published, context={"regime": "NORMAL"})
        pred = PredictionRecord(20, 0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "STATIC_v1", published)
        event_id = db.log_prediction(conn, event, pred)
        db.record_outcome(conn, event_id, magnitude, published + timedelta(days=20), magnitude - 0.02, "OK")
    baseline = estimate_baseline_from_ledger(conn, "GUIDANCE_CHANGE", [20])
    assert abs(baseline[20] - 0.15) < 1e-9  # mean(|0.10|, |0.20|)


def test_format_report_not_recognized():
    text = format_report({"status": "NOT_RECOGNIZED", "entity": "NVDA", "raw_text": "hello"})
    assert "SECURITY: NVDA" in text
    assert "NOT_RECOGNIZED" in text


def test_format_report_ok_contains_required_fields():
    conn = db.connect(":memory:")
    prices = FlatPrices()
    result = run_predict(conn, prices, "NVDA", "NVDA corp raises full-year guidance", "cli", PUBLISHED,
                          interpreter=RuleBasedInterpreter(), horizon_days_list=[1])
    text = format_report(result)
    for field in ("SECURITY:", "EVENT:", "DIRECTION:", "REGIME:", "EVIDENCE:"):
        assert field in text
    assert "1D:" in text
