from datetime import datetime, timezone

from market_agent.report import format_report
from market_agent.reporting.knowledge_state import build_knowledge_state_report
from market_agent.store import db

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_format_report_on_empty_ledger_does_not_crash():
    conn = db.connect(":memory:")
    report = build_knowledge_state_report(conn, now=NOW)
    text = format_report(report)
    assert "KNOWLEDGE STATE" in text
    assert "ACTIVE RELATIONSHIPS (0)" in text
    assert "OPERATIONAL COUNTS" in text
    assert "CALIBRATION BY HORIZON" in text


def test_format_report_includes_relationship_and_calibration_detail():
    conn = db.connect(":memory:")
    db.upsert_relationship(conn, "rel-1", {"event_type": "GUIDANCE_CHANGE", "direction": "negative"}, 20,
                            -0.09, -0.11, -0.07, 40, "ACTIVE", NOW, last_revalidated_at=NOW)
    report = build_knowledge_state_report(conn, now=NOW)
    text = format_report(report)
    assert "rel-1" in text
    assert "decay=FRESH" in text
