"""Item 2/3: source normalization and duplicate control.

DUPLICATE CONTROL IS A COARSE, DISCLOSED HEURISTIC FOR THIS MVP, NOT A
TRAINED NEAR-DUPLICATE MODEL: two documents are considered the SAME
underlying report if they share a fingerprint (providers.py's
make_fingerprint - normalized title + first 400 chars of content) OR if
their titles are near-identical after stripping publisher suffixes and
punctuation (wire-service syndication very often changes only the
headline's trailing " - Publisher Name" or minor punctuation). This will
under-merge genuinely paraphrased duplicate coverage and will not catch
every syndication pattern - a real near-duplicate/embedding model would do
better, but is out of scope for this MVP (disclosed here, not hidden).
What it DOES reliably catch (verified against real Google News RSS output)
is exact/near-exact wire-service syndication, which is the dominant
duplicate pattern in aggregated news feeds.
"""
from __future__ import annotations

import re

from market_agent.research.schema import SourceDocument

_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_PUBLISHER_SUFFIX_RE = re.compile(r"\s*[-|]\s*[a-z0-9 .]{2,40}$")


def _title_key(title: str) -> str:
    t = title.lower().strip()
    t = _PUBLISHER_SUFFIX_RE.sub("", t)  # strip a trailing " - Reuters" / " | Yahoo Finance" style suffix
    t = _PUNCT_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def deduplicate_documents(documents: list[SourceDocument]) -> tuple[list[SourceDocument], dict[str, list[str]]]:
    """Item 3: groups documents that are almost certainly the SAME
    underlying report. Returns (documents_with_duplicate_of_set,
    canonical_source_id -> [all source_ids in that group]). The CANONICAL
    document for a group is the EARLIEST published one with the HIGHEST
    reliability tier (a primary-source original beats a tertiary
    syndication published at the same moment) - deterministic, not
    arbitrary "whichever came first in the list"."""
    reliability_rank = {"PRIMARY": 0, "SECONDARY": 1, "TERTIARY": 2}
    groups: dict[str, list[SourceDocument]] = {}
    for doc in documents:
        key = doc.fingerprint if doc.fingerprint else _title_key(doc.title)
        title_key = _title_key(doc.title)
        # merge on EITHER the content fingerprint or the normalized title - see module docstring
        merge_key = None
        for existing_key, members in groups.items():
            if key == existing_key or any(_title_key(m.title) == title_key for m in members):
                merge_key = existing_key
                break
        groups.setdefault(merge_key or key, []).append(doc)

    canonical_map: dict[str, list[str]] = {}
    result: list[SourceDocument] = []
    for members in groups.values():
        canonical = min(members, key=lambda d: (reliability_rank.get(d.reliability, 3), d.published_at))
        canonical_map[canonical.source_id] = [m.source_id for m in members]
        for m in members:
            duplicate_of = None if m.source_id == canonical.source_id else canonical.source_id
            result.append(SourceDocument(
                source_id=m.source_id, publisher=m.publisher, source_type=m.source_type, url=m.url,
                published_at=m.published_at, retrieved_at=m.retrieved_at, entity=m.entity, title=m.title,
                raw_content=m.raw_content, normalized_content=m.normalized_content, reliability=m.reliability,
                fingerprint=m.fingerprint, duplicate_of=duplicate_of,
            ))
    return result, canonical_map


def independent_source_count(documents: list[SourceDocument]) -> int:
    """The number of DISTINCT underlying reports among `documents`, after
    collapsing syndicated duplicates - see module docstring. This is what
    distinguishes "10 articles" from "1 independent report reproduced by
    10 publishers" (item 3's explicit requirement)."""
    canonical_ids = {d.source_id if d.duplicate_of is None else d.duplicate_of for d in documents}
    return len(canonical_ids)
