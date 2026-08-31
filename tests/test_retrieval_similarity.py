from datetime import datetime, timezone

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.retrieval.similarity import find_similar_cases, prior_return_bucket
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_prior_return_bucket_boundaries():
    assert prior_return_bucket(-0.10) == "LARGE_DECLINE"
    assert prior_return_bucket(-0.01) == "FLAT"
    assert prior_return_bucket(0.10) == "LARGE_GAIN"
    assert prior_return_bucket(None) == "UNKNOWN"


def _log(conn, entity, regime, prior_return, published):
    event = EventRecord(entity=entity, event_type="GUIDANCE_CHANGE", direction="negative", source="wire",
                         source_reliability_snapshot=0.5, raw_text="cuts guidance", published_at=published,
                         ingested_at=published, context={"regime": regime, "prior_5d_return": prior_return})
    pred = PredictionRecord(20, -0.02, "MEDIUM", {"basis": "unconditional_baseline"}, "TEST_v1", published)
    event_id = db.log_prediction(conn, event, pred)
    db.record_outcome(conn, event_id, -0.03, published, -0.01, "OK")
    return event_id


def test_find_similar_cases_matches_regime_and_return_bucket():
    conn = db.connect(":memory:")
    _log(conn, "AAPL", "RISK_OFF", -0.09, datetime(2024, 1, 1, tzinfo=timezone.utc))   # matches
    _log(conn, "MSFT", "RISK_OFF", -0.09, datetime(2024, 2, 1, tzinfo=timezone.utc))   # matches
    _log(conn, "GOOG", "NORMAL", -0.09, datetime(2024, 3, 1, tzinfo=timezone.utc))     # wrong regime
    _log(conn, "AMZN", "RISK_OFF", 0.01, datetime(2024, 4, 1, tzinfo=timezone.utc))    # wrong return bucket

    cases = find_similar_cases(conn, "GUIDANCE_CHANGE", "RISK_OFF", -0.09, 20)
    assert {c.entity for c in cases} == {"AAPL", "MSFT"}


def test_find_similar_cases_respects_published_before():
    conn = db.connect(":memory:")
    _log(conn, "AAPL", "RISK_OFF", -0.09, datetime(2024, 1, 1, tzinfo=timezone.utc))
    _log(conn, "MSFT", "RISK_OFF", -0.09, datetime(2024, 12, 1, tzinfo=timezone.utc))
    cases = find_similar_cases(conn, "GUIDANCE_CHANGE", "RISK_OFF", -0.09, 20,
                                published_before="2024-06-01T00:00:00+00:00")
    assert {c.entity for c in cases} == {"AAPL"}
