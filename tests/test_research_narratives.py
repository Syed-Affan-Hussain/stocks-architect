from datetime import datetime, timedelta, timezone

from market_agent.research.narratives import cluster_narratives
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument, TimelineEvent

NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)


def _doc(source_id, days_ago, reliability="TERTIARY", duplicate_of=None):
    date = (NOW - timedelta(days=days_ago)).isoformat()
    return SourceDocument(source_id=source_id, publisher="Pub", source_type="NEWS", url="https://x",
                           published_at=date, retrieved_at=date, entity="ACME", title="t", raw_content="c",
                           normalized_content="c", reliability=reliability,
                           fingerprint=make_fingerprint("t", "c"), duplicate_of=duplicate_of)


def _event(event_id, source_id, days_ago, sentiment, event_type="GENERAL_NEWS", area="demand"):
    date = (NOW - timedelta(days=days_ago)).date().isoformat()
    return TimelineEvent(event_id=event_id, entity="ACME", date=date, event_type=event_type,
                          description="desc", evidence_type="REPORTING", source_ids=[source_id],
                          confidence="MEDIUM", materiality="MEDIUM", sentiment=sentiment, affected_area=area)


def test_events_with_same_type_and_area_cluster_into_one_narrative():
    events = [_event("e1", "d1", 2, "POSITIVE"), _event("e2", "d2", 3, "POSITIVE")]
    docs = [_doc("d1", 2), _doc("d2", 3)]
    narratives = cluster_narratives(events, docs, now=NOW)
    assert len(narratives) == 1
    assert narratives[0].source_count == 2


def test_independent_source_count_collapses_syndicated_duplicates():
    events = [_event("e1", "d1", 1, "POSITIVE"), _event("e2", "d2", 1, "POSITIVE"), _event("e3", "d3", 1, "POSITIVE")]
    docs = [_doc("d1", 1), _doc("d2", 1, duplicate_of="d1"), _doc("d3", 1, duplicate_of="d1")]
    narratives = cluster_narratives(events, docs, now=NOW)
    assert len(narratives) == 1
    assert narratives[0].source_count == 3
    assert narratives[0].independent_source_count == 1


def test_different_areas_produce_separate_narratives():
    events = [_event("e1", "d1", 1, "POSITIVE", area="demand"), _event("e2", "d2", 1, "NEGATIVE", area="margins")]
    docs = [_doc("d1", 1), _doc("d2", 1)]
    narratives = cluster_narratives(events, docs, now=NOW)
    assert len(narratives) == 2
    areas = {(n.description) for n in narratives}
    assert len(areas) == 2


def test_emerging_trend_when_only_recent_events_exist():
    events = [_event("e1", "d1", 2, "POSITIVE")]
    docs = [_doc("d1", 2)]
    narratives = cluster_narratives(events, docs, now=NOW)
    assert narratives[0].trend == "EMERGING"


def test_fading_trend_when_only_older_events_exist():
    events = [_event("e1", "d1", 18, "POSITIVE")]
    docs = [_doc("d1", 18)]
    narratives = cluster_narratives(events, docs, now=NOW)
    assert narratives[0].trend == "FADING"


def test_no_events_produces_no_narratives():
    assert cluster_narratives([], [], now=NOW) == []
