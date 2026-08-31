"""Real historical dividend-change events from SEC EDGAR's full-text
search - stage 5 item 3's first event-universe expansion beyond guidance
changes. Same proven sourcing pattern as sources/edgar_guidance.py
(reused deliberately, not reinvented - see events/interpret.py's module
docstring for why dividend changes were chosen as the next type: no
consensus-estimate/analyst-feed data source needed, unlike earnings
surprises or analyst actions).

VERIFIED, NOT ASSUMED - checked live against efts.sec.gov (2018-2024
8-K full-text search) before these phrase lists were written:

  increases quarterly dividend ....... 526 hits
  raises quarterly dividend ........... 70 hits
  increases its quarterly dividend .... 12 hits
  declares special dividend ........... 131 hits
  ---
  suspend its dividend ................ 14 hits
  suspends dividend .................... 8 hits
  eliminates dividend ................... 7 hits
  dividend suspension .................. 66 hits (used as a corroborating phrase, see CUT_PHRASES)
  will not pay a dividend .............. 42 hits
  reduces the quarterly dividend ....... 27 hits

THE SAME REAL, DISCLOSED ASYMMETRY AS GUIDANCE CHANGES: roughly 6x more
raw hits for dividend INCREASES than cuts/suspensions over this window.
This is a genuine property of corporate disclosure (companies rarely use
a sharp, formulaic verb near "dividend" when cutting one), not a bug in
this module or a gap in phrase coverage - the negative side of any
downstream experiment will hit hypothesis_testing's MIN_N gate harder,
same as GUIDANCE_CHANGE's own disclosed asymmetry. Expected, not
compensated for.

POINT-IN-TIME BY CONSTRUCTION: identical mechanism to edgar_guidance.py -
`enddt` passed to EDGAR itself, plus a local PointInTimeClock check as
defense-in-depth.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from market_agent.events.schema import RawItem
from market_agent.pit.clock import PointInTimeClock
from market_agent.sources.edgar_guidance import HEADERS, SEARCH_URL, _extract_ticker

# Deliberately reuses the SAME literal wording as events/interpret.py's DIVIDEND_RAISE_PATTERNS/
# DIVIDEND_CUT_PATTERNS regexes wherever it exists as real filing language - one taxonomy, not two
# divergent ones (same discipline as edgar_guidance.py's RAISE_PHRASES/CUT_PHRASES).
RAISE_PHRASES = ["increases quarterly dividend", "raises quarterly dividend", "increases its quarterly dividend",
                 "declares special dividend"]
CUT_PHRASES = ["suspend its dividend", "suspends dividend", "eliminates dividend", "dividend suspension",
               "will not pay a dividend", "reduces the quarterly dividend"]


@dataclass
class SourcedRawItem:
    raw_item: RawItem
    matched_phrase: str
    accession_number: str
    items_8k: list[str]


def _search_phrase(phrase: str, start_date: str, end_date: str, max_pages: int = 4) -> list[dict]:
    """Identical pagination/retry logic to edgar_guidance.py's private
    helper of the same name - not imported across that module's boundary
    since it's a tiny, self-contained HTTP loop with no shared state, and
    each source module should be independently readable end to end."""
    hits = []
    for page in range(max_pages):
        for attempt in range(3):
            resp = requests.get(SEARCH_URL, params={"q": f'"{phrase}"', "forms": "8-K",
                                                      "startdt": start_date, "enddt": end_date,
                                                      "from": page * 10},
                                 headers=HEADERS, timeout=20)
            body = resp.json() if resp.status_code == 200 else {}
            if "hits" in body:
                break
            time.sleep(2)
        else:
            break
        page_hits = body["hits"]["hits"]
        if not page_hits:
            break
        hits.extend(page_hits)
        time.sleep(0.3)
    return hits


def fetch_dividend_events(start_date: str, end_date: str, clock: PointInTimeClock,
                           cache_path: str | Path | None = None,
                           max_pages_per_phrase: int = 4) -> list[SourcedRawItem]:
    """Same one-time historical ingestion contract as
    edgar_guidance.py::fetch_guidance_events - cached to `cache_path` as
    JSON so a walk-forward experiment replaying this window doesn't
    re-hit the network."""
    if cache_path is not None and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text())
        return [_from_cached(c) for c in cached]

    seen_accessions: dict[str, set[str]] = {"positive": set(), "negative": set()}
    results: list[SourcedRawItem] = []
    for direction, phrases in (("positive", RAISE_PHRASES), ("negative", CUT_PHRASES)):
        for phrase in phrases:
            for hit in _search_phrase(phrase, start_date, end_date, max_pages=max_pages_per_phrase):
                src = hit["_source"]
                adsh = src["adsh"]
                if adsh in seen_accessions[direction]:
                    continue
                seen_accessions[direction].add(adsh)

                ticker = _extract_ticker(src["display_names"][0])
                if ticker is None:
                    continue
                file_date = datetime.strptime(src["file_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                clock.assert_not_future(file_date, label=f"EDGAR filing {adsh}")

                raw_item = RawItem(text=f"{src['display_names'][0]}: {phrase}", source="SEC EDGAR 8-K",
                                    entity=ticker, published_at=file_date)
                results.append(SourcedRawItem(raw_item=raw_item, matched_phrase=phrase,
                                               accession_number=adsh, items_8k=src.get("items", [])))

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps([_to_cacheable(r) for r in results], indent=2))
    return results


def _to_cacheable(r: SourcedRawItem) -> dict:
    return {"text": r.raw_item.text, "source": r.raw_item.source, "entity": r.raw_item.entity,
            "published_at": r.raw_item.published_at.isoformat(), "matched_phrase": r.matched_phrase,
            "accession_number": r.accession_number, "items_8k": r.items_8k}


def _from_cached(c: dict) -> SourcedRawItem:
    return SourcedRawItem(
        raw_item=RawItem(text=c["text"], source=c["source"], entity=c["entity"],
                          published_at=datetime.fromisoformat(c["published_at"])),
        matched_phrase=c["matched_phrase"], accession_number=c["accession_number"], items_8k=c["items_8k"],
    )
