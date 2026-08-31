"""Runs a MethodologyExtractor over RawMethodologySource entries and
writes the result to the store - trading_methodologies +
methodology_concept_links (store/schema.py). Mirrors learn/hypothesis.py's
formalize_and_store() shape: every extracted claim is written before
anything downstream ever uses it, so the full extraction result is always
auditable.

A methodology with ZERO extracted concept claims is still recorded (not
silently dropped) - a methodology the rule-based extractor's narrow
coverage failed to map to anything is itself a disclosed, visible fact
(see knowledge_state.py's methodology section), not something to hide.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

from market_agent.methodology.extractor import MethodologyExtractor
from market_agent.methodology.schema import RawMethodologySource
from market_agent.store import db


def ingest_methodology(conn: sqlite3.Connection, extractor: MethodologyExtractor, source: RawMethodologySource,
                        ingested_at: datetime) -> str:
    methodology_id = str(uuid.uuid4())
    db.add_methodology(conn, methodology_id, name=source.name, practitioner=source.practitioner,
                        source_type=source.source_type, source_description=source.raw_text,
                        extractor_name=extractor.NAME, ingested_at=ingested_at)
    for claim in extractor.extract(source):
        db.add_methodology_concept_link(conn, str(uuid.uuid4()), methodology_id, claim.concept.value,
                                         claim.rationale, ingested_at)
    return methodology_id


def ingest_corpus(conn: sqlite3.Connection, extractor: MethodologyExtractor, sources: list[RawMethodologySource],
                   ingested_at: datetime) -> list[str]:
    return [ingest_methodology(conn, extractor, source, ingested_at) for source in sources]
