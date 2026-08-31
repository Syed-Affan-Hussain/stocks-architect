"""Item 7: clusters TimelineEvents into broader Narratives.

RULE-BASED CLUSTERING FOR THIS MVP, DISCLOSED: events are grouped by
(event_type, affected_area) - the same underlying story ("AI demand
remains strong") told across many articles almost always shares BOTH an
event_type (e.g. GENERAL_NEWS/GUIDANCE_CHANGE) and an affected_area (e.g.
"demand"). This is coarser than true semantic clustering (an embedding
model would merge near-synonymous stories using different words for the
same theme) - see llm_synthesis.py for how a configured LLM can later
produce a richer narrative description ON TOP of this same grouping,
never replacing the underlying evidence grouping itself.

INDEPENDENT-SOURCE COUNTING USES normalize.py's DEDUP OUTPUT DIRECTLY -
ten events citing ten syndicated copies of the same wire story must count
as ONE independent source, not ten (item 3's requirement, carried through
to the narrative level here).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone

from market_agent.research.normalize import independent_source_count
from market_agent.research.schema import Narrative, SourceDocument, TimelineEvent

RECENT_WINDOW_DAYS = 7   # "latest" window for trend comparison - fixed, disclosed
EARLIER_WINDOW_DAYS = 14  # the window immediately before the recent one


def _narrative_key(event: TimelineEvent) -> tuple[str, str | None]:
    return (event.event_type, event.affected_area)


def _describe(key: tuple[str, str | None], events: list[TimelineEvent]) -> str:
    event_type, area = key
    area_label = (area or "general").replace("_", " ")
    sentiments = Counter(e.sentiment for e in events)
    dominant = sentiments.most_common(1)[0][0]
    verb = {"POSITIVE": "improving", "NEGATIVE": "deteriorating", "MIXED": "mixed", "NEUTRAL": "developing"}[dominant]
    return f"{area_label.capitalize()}-related coverage ({event_type}) is {verb} based on {len(events)} reported event(s)."


def _dominant_sentiment(events: list[TimelineEvent]) -> str:
    counts = Counter(e.sentiment for e in events)
    counts.pop("NEUTRAL", None)
    if not counts:
        return "NEUTRAL"
    if len(counts) > 1 and len(set(counts.values())) == 1:
        return "MIXED"  # a genuine tie between positive/negative framings
    return counts.most_common(1)[0][0]


def _confidence_for(independent_sources: int, source_quality: str) -> str:
    if independent_sources >= 3 and source_quality in ("PRIMARY", "SECONDARY"):
        return "HIGH"
    if independent_sources >= 2:
        return "MEDIUM"
    return "LOW"


def _dominant_reliability(events: list[TimelineEvent], docs_by_id: dict[str, SourceDocument]) -> str:
    tiers = [docs_by_id[sid].reliability for e in events for sid in e.source_ids if sid in docs_by_id]
    if not tiers:
        return "MIXED"
    rank = {"PRIMARY": 0, "SECONDARY": 1, "TERTIARY": 2}
    return min(tiers, key=lambda t: rank.get(t, 3))


def _classify_trend(events: list[TimelineEvent], now: datetime) -> str:
    """Item 7's six trend states, from a simple, disclosed comparison of
    recent-window vs. earlier-window event counts and sentiment."""
    recent_cutoff = now - timedelta(days=RECENT_WINDOW_DAYS)
    earlier_cutoff = now - timedelta(days=RECENT_WINDOW_DAYS + EARLIER_WINDOW_DAYS)

    def _in_window(e, start, end):
        try:
            d = datetime.fromisoformat(e.date).replace(tzinfo=timezone.utc) if "T" not in e.date else \
                datetime.fromisoformat(e.date)
        except ValueError:
            return False
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return start <= d <= end

    recent = [e for e in events if _in_window(e, recent_cutoff, now)]
    earlier = [e for e in events if _in_window(e, earlier_cutoff, recent_cutoff)]

    if not earlier and recent:
        return "EMERGING"
    if not recent and earlier:
        return "FADING"
    if not recent and not earlier:
        return "STABLE"  # only older events exist, no recent activity either way

    recent_sent = _dominant_sentiment(recent)
    earlier_sent = _dominant_sentiment(earlier)
    if recent_sent == "MIXED" or (recent_sent != earlier_sent and recent_sent != "NEUTRAL"
                                   and earlier_sent != "NEUTRAL" and recent_sent != earlier_sent):
        return "DISPUTED" if recent_sent == "MIXED" else "STRENGTHENING" if len(recent) > len(earlier) else "WEAKENING"
    if len(recent) > len(earlier) * 1.3:
        return "STRENGTHENING"
    if len(recent) < len(earlier) * 0.7:
        return "WEAKENING"
    return "STABLE"


def cluster_narratives(events: list[TimelineEvent], documents: list[SourceDocument],
                        now: datetime | None = None) -> list[Narrative]:
    now = now or datetime.now(timezone.utc)
    docs_by_id = {d.source_id: d for d in documents}
    groups: dict[tuple[str, str | None], list[TimelineEvent]] = {}
    for event in events:
        groups.setdefault(_narrative_key(event), []).append(event)

    narratives: list[Narrative] = []
    for key, group_events in groups.items():
        if not group_events:
            continue
        entity = group_events[0].entity
        all_source_ids = [sid for e in group_events for sid in e.source_ids]
        contributing_docs = [docs_by_id[sid] for sid in all_source_ids if sid in docs_by_id]
        independent = independent_source_count(contributing_docs) if contributing_docs else len(set(all_source_ids))
        dominant_sentiment = _dominant_sentiment(group_events)
        source_quality = _dominant_reliability(group_events, docs_by_id)
        dates = sorted(e.date for e in group_events)
        narrative_id = "N_" + hashlib.sha256(f"{entity}:{key}".encode()).hexdigest()[:12]

        # MIXED sits with (POSITIVE, NEUTRAL) here, not in its own bucket - the alternative (dropping
        # it from both lists) was a real bug: an event with no assigned bucket is unreachable from
        # narrative.supporting_event_ids + narrative.contradicting_event_ids, which silently drops it
        # everywhere downstream that walks a narrative's constituent events (found via news_state/
        # event_vector.py's build_event_vectors producing ZERO EventVectors for a real MIXED-sentiment
        # clause - every event must land in exactly one of these two lists).
        supporting = [e.event_id for e in group_events if e.sentiment in ("POSITIVE", "NEUTRAL", "MIXED")]
        contradicting = [e.event_id for e in group_events if e.sentiment == "NEGATIVE"]
        if dominant_sentiment == "NEGATIVE":
            supporting, contradicting = contradicting, supporting

        narratives.append(Narrative(
            narrative_id=narrative_id, entity=entity, description=_describe(key, group_events),
            affected_area=key[1], supporting_event_ids=supporting, contradicting_event_ids=contradicting,
            first_observed=dates[0], latest_update=dates[-1], source_count=len(all_source_ids),
            independent_source_count=independent, source_quality=source_quality, sentiment=dominant_sentiment,
            confidence=_confidence_for(independent, source_quality), trend=_classify_trend(group_events, now),
        ))
    narratives.sort(key=lambda n: (-n.source_count, n.narrative_id))
    return narratives
