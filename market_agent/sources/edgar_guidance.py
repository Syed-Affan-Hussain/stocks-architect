"""Real historical guidance-change events from SEC EDGAR's full-text
search - Blueprint stage: "real historical guidance-change event source
using the existing point-in-time EDGAR approach".

VERIFIED, NOT ASSUMED (checked live against efts.sec.gov before writing
this module): EDGAR full-text search returns, per hit, `_source.file_date`
(the actual FILING date - the authoritative point-in-time timestamp for
this system's purposes, never the fiscal period the filing describes),
`_source.display_names` (a "COMPANY NAME  (TICKER)  (CIK ...)" string a
ticker can be pulled out of directly, with no separate CIK->ticker map
fetch needed), and `_source.form`/`items` (8-K item codes).

PHRASE COVERAGE IS REAL BUT ASYMMETRIC - disclosed, not hidden: querying
exact phrases live against 2018-2024 filings found roughly 4-8x more raw
hits for guidance RAISES than CUTS (companies phrase negative guidance
changes with much more varied, softer language - "revises", "updates",
generic language, or bury it inside earnings-release prose without a
sharp verb near the word "guidance" at all - as opposed to the more
formulaic "raises guidance" wording common on the upside). This is a
genuine property of how companies communicate, not a bug in this module,
and it means the CUT side of any downstream experiment will have
materially less statistical power - expect it to hit hypothesis_testing's
MIN_N gate more often, which is the correct, honest outcome rather than
something to compensate for by inventing more cut-side phrases until the
sample sizes look symmetric.

POINT-IN-TIME BY CONSTRUCTION, not by a post-hoc filter: `enddt` is
passed to EDGAR itself, so the search only ever returns filings that
existed as of that cutoff - this system never fetches "everything" and
filters locally. A local PointInTimeClock check is still applied as
defense-in-depth (see fetch_guidance_events), consistent with this
project's established belt-and-suspenders pattern for leakage risk.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from market_agent.events.schema import RawItem
from market_agent.pit.clock import PointInTimeClock

HEADERS = {"User-Agent": "Stocks_Architect research (contact: affanhussain2003@gmail.com)"}
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Curated from a live check against efts.sec.gov (see module docstring) - not guessed at.
# Deliberately reuses the SAME literal wording as market_agent.events.interpret's regex
# patterns wherever it exists as real filing language, so a hit here is classified
# identically to how RuleBasedInterpreter would classify locally-sourced text containing
# the same phrase - one taxonomy, not two divergent ones.
RAISE_PHRASES = ["raises guidance", "raised guidance", "increases guidance", "raises outlook",
                 "raises full-year guidance"]
CUT_PHRASES = ["lowers guidance", "lowered guidance", "reduces guidance", "lowers outlook",
               "withdraws guidance", "lowers full-year guidance", "lowers full-year outlook"]

TICKER_PATTERN = re.compile(r"\(([A-Z]{1,6})\)")


@dataclass
class SourcedRawItem:
    raw_item: RawItem
    matched_phrase: str
    accession_number: str
    items_8k: list[str]


def _extract_ticker(display_name: str) -> str | None:
    match = TICKER_PATTERN.search(display_name)
    return match.group(1) if match else None


def _search_phrase(phrase: str, start_date: str, end_date: str, max_pages: int = 4) -> list[dict]:
    """Paginates through EDGAR FTS for one exact phrase. EDGAR's `from`
    parameter pages results 10 at a time up to its own cap; `max_pages`
    bounds how much of a very common phrase's results this module will
    pull in one run, not how far EDGAR itself allows paging."""
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
            break  # gave up on this page after 3 failed attempts - move on rather than hang
        page_hits = body["hits"]["hits"]
        if not page_hits:
            break
        hits.extend(page_hits)
        time.sleep(0.3)  # courteous pacing, well under SEC's stated rate limit
    return hits


def fetch_guidance_events(start_date: str, end_date: str, clock: PointInTimeClock,
                           cache_path: str | Path | None = None, max_pages_per_phrase: int = 4) -> list[SourcedRawItem]:
    """One-time historical ingestion: real EDGAR 8-K hits for
    RAISE_PHRASES/CUT_PHRASES between start_date and end_date (both
    'YYYY-MM-DD'), deduplicated by accession number within each
    direction. Cached to `cache_path` as JSON so a walk-forward
    experiment that replays this same window many times doesn't
    re-hit the network - the network call happens once, ever, per
    (start_date, end_date) pair; everything downstream (the walk-forward
    harness) filters this already-fetched, already-in-the-past dataset
    locally by `published_at`, which is the correct point-in-time
    simulation pattern (ingest real history once, replay many times)."""
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
                    continue  # same filing already matched a different phrase for this direction
                seen_accessions[direction].add(adsh)

                ticker = _extract_ticker(src["display_names"][0])
                if ticker is None:
                    continue  # cannot attribute to an entity - correctly excluded, not guessed at
                file_date = datetime.strptime(src["file_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                clock.assert_not_future(file_date, label=f"EDGAR filing {adsh}")  # defense-in-depth

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
