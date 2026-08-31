"""Network-free tests for sources/edgar_dividend.py - same testing
boundary as the rest of this project's EDGAR sourcing (the live HTTP
paths are exercised by the real-data experiment scripts, not the unit
test suite; see sources/edgar_guidance.py, which has no dedicated test
file either). What IS unit-tested here: the cache read/write round-trip,
which is the actual code path every downstream test/experiment run uses
once data has been fetched once.
"""
import json
from datetime import datetime, timezone

from market_agent.events.schema import RawItem
from market_agent.sources.edgar_dividend import SourcedRawItem, _from_cached, _to_cacheable, fetch_dividend_events

PUBLISHED = datetime(2024, 3, 15, tzinfo=timezone.utc)


def _item():
    return SourcedRawItem(
        raw_item=RawItem(text="ACME CORP (ACME): increases quarterly dividend", source="SEC EDGAR 8-K",
                          entity="ACME", published_at=PUBLISHED),
        matched_phrase="increases quarterly dividend", accession_number="acc-1", items_8k=["8.01"])


def test_cacheable_round_trip_preserves_all_fields():
    original = _item()
    restored = _from_cached(_to_cacheable(original))
    assert restored.raw_item.text == original.raw_item.text
    assert restored.raw_item.entity == original.raw_item.entity
    assert restored.raw_item.published_at == original.raw_item.published_at
    assert restored.matched_phrase == original.matched_phrase
    assert restored.accession_number == original.accession_number
    assert restored.items_8k == original.items_8k


def test_fetch_dividend_events_reads_from_cache_without_network(tmp_path):
    cache_path = tmp_path / "dividend_cache.json"
    cache_path.write_text(json.dumps([_to_cacheable(_item())]))

    from market_agent.pit.clock import PointInTimeClock
    clock = PointInTimeClock(now=datetime(2025, 1, 1, tzinfo=timezone.utc))
    results = fetch_dividend_events("2018-01-01", "2024-12-31", clock, cache_path=cache_path)

    assert len(results) == 1
    assert results[0].raw_item.entity == "ACME"
    assert results[0].matched_phrase == "increases quarterly dividend"


def test_raise_and_cut_phrases_are_disjoint():
    from market_agent.sources.edgar_dividend import CUT_PHRASES, RAISE_PHRASES
    assert set(RAISE_PHRASES).isdisjoint(set(CUT_PHRASES))
    assert len(RAISE_PHRASES) > 0 and len(CUT_PHRASES) > 0
