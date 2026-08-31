from market_agent.research.news_state.event_identity import cluster_events
from market_agent.research.news_state.magnitude import MagnitudeFact
from market_agent.research.schema import TimelineEvent


def _event(event_id, date, event_type="GENERAL_NEWS", area="revenue", entity="ACME"):
    return TimelineEvent(event_id=event_id, entity=entity, date=date, event_type=event_type,
                          description="d", evidence_type="REPORTING", source_ids=["s"], confidence="MEDIUM",
                          materiality="MEDIUM", sentiment="POSITIVE", affected_area=area)


def test_same_topic_within_window_merges():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-02"), _event("e3", "2024-06-03")]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_same_topic_outside_window_splits():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-20")]  # 19 days apart
    clusters = cluster_events(events)
    assert len(clusters) == 2


def test_different_topics_never_merge_even_same_day():
    events = [_event("e1", "2024-06-01", area="revenue"), _event("e2", "2024-06-01", area="regulatory")]
    clusters = cluster_events(events)
    assert len(clusters) == 2


def test_different_entities_never_merge():
    events = [_event("e1", "2024-06-01", entity="ACME"), _event("e2", "2024-06-01", entity="OTHER")]
    clusters = cluster_events(events)
    assert len(clusters) == 2


def test_magnitude_divergence_splits_same_topic_same_window():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-02")]
    magnitudes = {
        "e1": MagnitudeFact(raw_text="2%", value=2.0, unit="PERCENT"),
        "e2": MagnitudeFact(raw_text="40%", value=40.0, unit="PERCENT"),
    }
    clusters = cluster_events(events, magnitudes)
    assert len(clusters) == 2


def test_similar_magnitudes_within_tolerance_still_merge():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-02")]
    magnitudes = {
        "e1": MagnitudeFact(raw_text="40%", value=40.0, unit="PERCENT"),
        "e2": MagnitudeFact(raw_text="42%", value=42.0, unit="PERCENT"),
    }
    clusters = cluster_events(events, magnitudes)
    assert len(clusters) == 1


def test_missing_magnitude_on_either_side_defaults_to_merging():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-02")]
    magnitudes = {"e1": MagnitudeFact(raw_text="40%", value=40.0, unit="PERCENT")}  # e2 has none
    clusters = cluster_events(events, magnitudes)
    assert len(clusters) == 1  # no evidence of divergence -> conservative merge


def test_chain_transitivity_extends_a_cluster_beyond_one_direct_pairwise_gap():
    """A DOCUMENTED, real property of union-find clustering: e1 and e3
    are 6 days apart (outside TIME_WINDOW_DAYS=4 on their own), but each
    is within 3 days of e2 - continuous coverage with no gap correctly
    extends the SAME event's cluster, matching "an evolving story with
    ongoing coverage" rather than artificially capping it at one pairwise
    hop. A genuine gap (no intermediate coverage) still splits normally -
    see test_same_topic_outside_window_splits."""
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-04"), _event("e3", "2024-06-07")]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_three_way_chain_within_window_all_merge():
    events = [_event("e1", "2024-06-01"), _event("e2", "2024-06-03"), _event("e3", "2024-06-05")]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3
