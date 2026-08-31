"""Event identity - distinguishes separate real-world occurrences that
happen to share a topic, while still merging multiple articles that
report the SAME occurrence.

WHY THIS REPLACES narratives.py's CLUSTERING FOR THE NEWS-STATE PATH ONLY:
narratives.py (research/narratives.py) groups every clause sharing
(event_type, affected_area) into one Narrative, for the WHOLE collection
window, with no further check - the original v1 of this design reused
that directly. It is untouched and still correct for its own consumers
(consistency.py, risk.py, report_format.py, change_detection.py all still
use it exactly as before). For the news-state event quantifier
specifically, topic alone is not enough: two unrelated regulatory stories
two weeks apart would merge into one blob, and a single earnings story
covered for three days would still merge correctly by luck, not by
design. This module adds the two REAL, disclosed signals that make event
identity defensible instead of accidental:

  1. TIME WINDOW - clauses more than TIME_WINDOW_DAYS apart, even on the
     SAME topic, are never merged. This is deliberately DAY-granularity,
     matching what TimelineEvent.date actually carries (extraction.py
     truncates a document's published_at to a bare date - there is no
     time-of-day resolution anywhere upstream to be more precise than
     this without fabricating precision that doesn't exist).

  2. MAGNITUDE DIVERGENCE - when BOTH sides of a same-topic, same-window
     pair have an extracted numeric magnitude (magnitude.py) for what
     looks like the same claim, and those magnitudes disagree by more
     than a disclosed relative tolerance, they are treated as two
     different facts (a genuine revision, or two different metrics),
     never blindly averaged into one. Absent a magnitude on EITHER side,
     this signal is silent (no evidence of divergence -> default to
     merging within the topic+window, the more conservative choice given
     no textual-similarity check is reliable enough to split on - see
     below).

TEXT SIMILARITY IS DELIBERATELY NOT A HARD SPLITTING SIGNAL: the earlier
validation (Experiment D) measured REAL same-event paraphrases scoring as
low as 0.12 TF-IDF cosine similarity - a similarity THRESHOLD low enough
to keep true paraphrases together would do almost no useful splitting;
one high enough to split genuinely different stories would also
incorrectly split true paraphrases. It remains available
(text_similarity.py) as a diagnostic/display signal, never as a merge/
split decision here.
"""
from __future__ import annotations

from datetime import datetime

from market_agent.research.news_state.magnitude import MagnitudeFact
from market_agent.research.schema import TimelineEvent

TIME_WINDOW_DAYS = 4.0  # fixed, disclosed - approximates a single news-cycle's follow-up coverage
#                          window (initial report + a few days of analysis/reaction pieces), matching
#                          the same order of magnitude as aggregation.py's 7-day half-life without
#                          being the same number for a different reason - this bounds CLUSTERING,
#                          that bounds AGGREGATION WEIGHT.
MAGNITUDE_DIVERGENCE_RATIO = 0.5  # fixed, disclosed - a >50% relative difference between two
#                                    same-topic, same-window extracted magnitudes is treated as
#                                    evidence of two different facts, not noisy reporting of one.


def _topic_key(event: TimelineEvent) -> tuple:
    return (event.entity, event.event_type, event.affected_area)


def _days_apart(a: TimelineEvent, b: TimelineEvent) -> float:
    try:
        da, db = datetime.fromisoformat(a.date), datetime.fromisoformat(b.date)
    except ValueError:
        return 0.0
    return abs((da - db).days)


def _magnitudes_diverge(a: TimelineEvent, b: TimelineEvent, magnitudes_by_event_id: dict[str, MagnitudeFact]
                         ) -> bool:
    ma, mb = magnitudes_by_event_id.get(a.event_id), magnitudes_by_event_id.get(b.event_id)
    if ma is None or mb is None or ma.unit != mb.unit:
        return False  # no evidence either way -> not treated as divergent (conservative default)
    denom = max(abs(ma.value), abs(mb.value), 1e-9)
    return abs(ma.value - mb.value) / denom > MAGNITUDE_DIVERGENCE_RATIO


def _connected_components(group: list[TimelineEvent], magnitudes_by_event_id: dict[str, MagnitudeFact]
                           ) -> list[list[TimelineEvent]]:
    n = len(group)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _days_apart(group[i], group[j]) <= TIME_WINDOW_DAYS and \
                    not _magnitudes_diverge(group[i], group[j], magnitudes_by_event_id):
                union(i, j)

    components: dict[int, list[TimelineEvent]] = {}
    for idx, event in enumerate(group):
        components.setdefault(find(idx), []).append(event)
    return list(components.values())


def cluster_events(events: list[TimelineEvent],
                    magnitudes_by_event_id: dict[str, MagnitudeFact] | None = None) -> list[list[TimelineEvent]]:
    """Groups TimelineEvents into real event-instance clusters: same
    topic AND within TIME_WINDOW_DAYS of each other AND (when both sides
    have a comparable extracted magnitude) not magnitude-divergent. Each
    returned list is one candidate real-world occurrence - build_event_
    vector (event_vector.py) turns each into one EventVector."""
    magnitudes_by_event_id = magnitudes_by_event_id or {}
    topic_groups: dict[tuple, list[TimelineEvent]] = {}
    for e in events:
        topic_groups.setdefault(_topic_key(e), []).append(e)

    clusters: list[list[TimelineEvent]] = []
    for group in topic_groups.values():
        clusters.extend(_connected_components(group, magnitudes_by_event_id))
    return clusters
