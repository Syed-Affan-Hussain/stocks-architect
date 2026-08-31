from datetime import datetime, timedelta, timezone

from market_agent.research.news_state.pipeline import build_news_state_from_documents
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument
from market_agent.store import db

NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)


def _doc(source_id, title, content, publisher="Pub", reliability="TERTIARY", days_ago=0):
    date = (NOW - timedelta(days=days_ago)).isoformat()
    return SourceDocument(source_id=source_id, publisher=publisher, source_type="NEWS", url=f"https://x/{source_id}",
                           published_at=date, retrieved_at=date, entity="ACME", title=title, raw_content=content,
                           normalized_content=content, reliability=reliability,
                           fingerprint=make_fingerprint(title, content))


def test_end_to_end_build_produces_a_real_state():
    conn = db.connect(":memory:")
    docs = [
        _doc("d1", "ACME reports stronger-than-expected revenue growth",
             "ACME reported stronger-than-expected revenue growth this quarter, while management raised its "
             "full-year guidance.", publisher="Reuters", reliability="SECONDARY"),
        _doc("d2", "ACME revenue beats expectations",
             "ACME revenue beat expectations, and the company raised its outlook for the year.",
             publisher="Yahoo Finance"),
    ]
    state, event_vectors = build_news_state_from_documents("ACME", docs, conn, as_of=NOW)
    assert state.entity == "ACME"
    assert state.news_volume == 2
    assert event_vectors  # at least one real EventVector produced
    assert state.dimensions["growth"] is not None
    assert state.dimensions["growth"] > 0


def test_persistence_round_trip_enables_trajectory_on_second_call():
    conn = db.connect(":memory:")
    docs_t1 = [_doc("d1", "ACME revenue declines",
                     "ACME reported that revenue declined sharply this quarter.", days_ago=5)]
    state1, _ = build_news_state_from_documents("ACME", docs_t1, conn, as_of=NOW - timedelta(days=5))

    docs_t2 = [_doc("d2", "ACME revenue rebounds",
                     "ACME reported that revenue increased strongly this quarter.", days_ago=0)]
    state2, _ = build_news_state_from_documents("ACME", docs_t2, conn, as_of=NOW)

    assert state2.state_change is not None
    assert state2.state_change.get("growth", 0) > 0  # moved from negative-leaning to positive
    assert state2.state_velocity is not None


def test_no_documents_produces_an_empty_but_valid_state():
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("ACME", [], conn, as_of=NOW)
    assert event_vectors == []
    assert state.news_volume == 0
    assert all(v is None for v in state.dimensions.values())
