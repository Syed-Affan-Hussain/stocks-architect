"""Orchestrates the full article -> event -> company-state pipeline,
reusing research/'s EXISTING normalize/extract machinery (dedup, clause
splitting) while using event_identity.py's OWN, stricter event-instance
clustering (topic + time window + magnitude divergence) rather than
research/narratives.py's purely topical grouping - see event_identity.py's
module docstring for exactly why the news-state path needed its own
mechanism.

ATTRIBUTION GATE (entity_resolution.py, added after Event Quantifier
v1.0): before clustering/scoring, every extracted clause is checked for
WHO it's actually about - a clause attributed to a named competitor or to
the industry at large (not the queried entity, and not a counterparty/
subsidiary of it) is excluded here, before event_identity.py or
event_vector.py ever see it. This is a FILTER on which clauses are
eligible, not a change to how an eligible clause is scored - v1.0's
scoring formulas (magnitude anchors, modality weights, contradiction
threshold) are unchanged.

TWO ENTRY POINTS, DELIBERATELY SEPARATE:

`build_news_state_from_documents` takes an already-collected list of
SourceDocument and runs the full dedup -> extract -> cluster -> event-
vector -> aggregate -> persist chain. This is the one the controlled
validation experiments use, with hand-authored SourceDocument lists, so
each experiment can isolate exactly one property (duplication,
contradiction, paraphrase, ...) without depending on what real news
happens to be live right now.

`fetch_and_compute_news_state` is the live convenience wrapper - fetches
real Google News RSS via research/providers.py's NewsProvider, then calls
the function above. This is what the 5-company real-data validation uses.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from market_agent.research.extraction import extract_all_events
from market_agent.research.news_state.aggregation import (
    aggregate_company_state, compute_novelty, persist_company_state, persist_event_vectors,
)
from market_agent.research.news_state.entity_resolution import resolve_entity_role
from market_agent.research.news_state.event_identity import cluster_events
from market_agent.research.news_state.event_vector import build_event_vectors
from market_agent.research.news_state.magnitude import extract_primary_magnitude
from market_agent.research.news_state.schema import CompanyNewsState, EventVector
from market_agent.research.normalize import deduplicate_documents
from market_agent.research.providers import NewsProvider, SourceDocument
from market_agent.store import db


def _state_from_row(row: sqlite3.Row) -> CompanyNewsState:
    import json
    d = json.loads(row["state_json"])
    return CompanyNewsState(
        entity=d["entity"], as_of=d["as_of"], dimensions=d["dimensions"], dispersion=d["dispersion"],
        text_sentiment=d.get("text_sentiment"), confidence=d.get("confidence", 0.0),
        news_volume=d.get("news_volume", 0), independent_event_count=d.get("independent_event_count", 0),
        dominant_event_ids=d.get("dominant_event_ids", []), contradiction_axes=d.get("contradiction_axes", []),
        source_quality=d.get("source_quality", 0.0), state_change=d.get("state_change"),
        state_velocity=d.get("state_velocity"), state_direction=d.get("state_direction"),
        half_life_days=d.get("half_life_days", 7.0), excluded_by_role=d.get("excluded_by_role", {}),
    )


def build_news_state_from_documents(entity: str, documents: list[SourceDocument], conn: sqlite3.Connection,
                                     as_of: datetime | None = None, persist: bool = True,
                                     entity_aliases: tuple[str, ...] = ()
                                     ) -> tuple[CompanyNewsState, list[EventVector]]:
    """`entity_aliases`: name variants for `entity` (e.g. its registered
    company name, since real news almost always uses that rather than the
    ticker) - passed through to entity_resolution.resolve_entity_role so
    the attribution gate can recognize the entity being named even when
    the raw `entity` string never literally appears. Optional and empty
    by default - see entity_resolution.py's module docstring for what
    degrades (not breaks) without it."""
    as_of = as_of or datetime.now(timezone.utc)
    raw_document_count = len(documents)

    deduped_documents, _canonical_map = deduplicate_documents(documents)
    documents_by_id = {d.source_id: d for d in deduped_documents}

    all_events = extract_all_events(deduped_documents)

    # --- ATTRIBUTION GATE: exclude clauses about a different real-world party before anything below
    # ever sees them - see entity_resolution.py and this module's docstring. ---
    events: list = []
    excluded_by_role: dict[str, int] = {}
    for e in all_events:
        args = resolve_entity_role(e.description, entity, entity_aliases)
        if args.attributable:
            events.append(e)
        else:
            excluded_by_role[args.subject_role] = excluded_by_role.get(args.subject_role, 0) + 1

    magnitudes_by_event_id = {}
    for e in events:
        fact = extract_primary_magnitude(e.description)
        if fact is not None:
            magnitudes_by_event_id[e.event_id] = fact

    clusters = cluster_events(events, magnitudes_by_event_id)
    event_vectors = build_event_vectors(entity, clusters, documents_by_id)

    for ev in event_vectors:
        ev.novelty = compute_novelty(conn, ev)

    prior_row = db.latest_news_company_state(conn, entity, before=as_of.isoformat())
    prior_state = _state_from_row(prior_row) if prior_row is not None else None

    state = aggregate_company_state(entity, event_vectors, as_of, raw_document_count=raw_document_count,
                                     prior_state=prior_state)
    state.excluded_by_role = excluded_by_role

    if persist:
        persist_event_vectors(conn, event_vectors, computed_at=as_of)
        persist_company_state(conn, state, computed_at=as_of)

    return state, event_vectors


def fetch_and_compute_news_state(entity: str, conn: sqlite3.Connection, max_items: int = 40,
                                  as_of: datetime | None = None, persist: bool = True,
                                  entity_aliases: tuple[str, ...] = ()
                                  ) -> tuple[CompanyNewsState, list[EventVector], list[SourceDocument]]:
    as_of = as_of or datetime.now(timezone.utc)
    result = NewsProvider().fetch(entity, entity, max_items=max_items)
    state, event_vectors = build_news_state_from_documents(entity, result.documents, conn, as_of, persist,
                                                             entity_aliases)
    return state, event_vectors, result.documents
