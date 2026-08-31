from market_agent.research.normalize import deduplicate_documents, independent_source_count
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument


def _doc(source_id, title, content, publisher="Pub", reliability="TERTIARY", published_at="2024-01-01T00:00:00+00:00"):
    return SourceDocument(source_id=source_id, publisher=publisher, source_type="NEWS", url=f"https://x/{source_id}",
                           published_at=published_at, retrieved_at=published_at, entity="ACME", title=title,
                           raw_content=content, normalized_content=content, reliability=reliability,
                           fingerprint=make_fingerprint(title, content))


def test_identical_fingerprint_is_merged_into_one_group():
    docs = [_doc("a", "Acme reports record revenue", "Acme Corp today reported record revenue for Q3."),
            _doc("b", "Acme reports record revenue", "Acme Corp today reported record revenue for Q3.")]
    result, canonical_map = deduplicate_documents(docs)
    assert independent_source_count(result) == 1
    assert len(canonical_map) == 1


def test_syndicated_title_with_different_publisher_suffix_is_merged():
    docs = [
        _doc("a", "Acme reports record revenue - Reuters", "Acme Corp today reported record revenue.", publisher="Reuters", reliability="SECONDARY"),
        _doc("b", "Acme reports record revenue | Yahoo Finance", "Different wording of the same wire story.", publisher="Yahoo Finance"),
        _doc("c", "Acme reports record revenue - Business Insider", "Yet another republication.", publisher="Business Insider"),
    ]
    result, canonical_map = deduplicate_documents(docs)
    assert independent_source_count(result) == 1
    # the PRIMARY/SECONDARY-ranked, earliest original should be the canonical one
    canonical_ids = list(canonical_map.keys())
    assert canonical_ids[0] == "a"


def test_genuinely_different_stories_are_not_merged():
    docs = [_doc("a", "Acme reports record revenue", "Revenue growth story."),
            _doc("b", "Acme CEO resigns amid controversy", "Leadership change story.")]
    result, canonical_map = deduplicate_documents(docs)
    assert independent_source_count(result) == 2
    assert len(canonical_map) == 2


def test_duplicate_of_is_none_for_the_canonical_document():
    docs = [_doc("a", "Same story", "content"), _doc("b", "Same story", "content")]
    result, _ = deduplicate_documents(docs)
    canonical = [d for d in result if d.duplicate_of is None]
    duplicates = [d for d in result if d.duplicate_of is not None]
    assert len(canonical) == 1
    assert len(duplicates) == 1
    assert duplicates[0].duplicate_of == canonical[0].source_id


def test_independent_source_count_on_empty_list():
    assert independent_source_count([]) == 0
