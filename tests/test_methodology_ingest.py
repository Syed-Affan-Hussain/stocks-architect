from datetime import datetime, timezone

from market_agent.methodology.extractor import RuleBasedMethodologyExtractor
from market_agent.methodology.ingest import ingest_corpus, ingest_methodology
from market_agent.methodology.schema import RawMethodologySource
from market_agent.methodology.seed_corpus import SEED_CORPUS
from market_agent.store import db

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_ingest_methodology_writes_methodology_and_concept_links():
    conn = db.connect(":memory:")
    source = RawMethodologySource(name="Test System", practitioner="Test Trader", source_type="book",
                                   raw_text="A breakout system confirmed by a volume surge.")
    methodology_id = ingest_methodology(conn, RuleBasedMethodologyExtractor(), source, NOW)

    row = db.get_methodology(conn, methodology_id)
    assert row["name"] == "Test System"
    assert row["extractor_name"] == "RULE_BASED"

    links = db.concept_links_for_methodology(conn, methodology_id)
    concepts = {r["concept"] for r in links}
    assert "BREAKOUT" in concepts
    assert "VOLUME" in concepts


def test_ingest_methodology_with_zero_matches_is_still_recorded():
    conn = db.connect(":memory:")
    source = RawMethodologySource(name="Unrelated", practitioner="X", source_type="book",
                                   raw_text="A discussion of quarterly earnings calls with no technical content.")
    methodology_id = ingest_methodology(conn, RuleBasedMethodologyExtractor(), source, NOW)
    assert db.get_methodology(conn, methodology_id) is not None
    assert db.concept_links_for_methodology(conn, methodology_id) == []


def test_ingest_corpus_ingests_every_source():
    conn = db.connect(":memory:")
    ids = ingest_corpus(conn, RuleBasedMethodologyExtractor(), SEED_CORPUS, NOW)
    assert len(ids) == len(SEED_CORPUS)
    assert len(db.all_methodologies(conn)) == len(SEED_CORPUS)


def test_seed_corpus_produces_multiple_methodologies_contributing_to_the_same_concept():
    """The 'multiple independent methodologies reinforce the same concept'
    scenario, proven against the real seed corpus rather than a synthetic
    fixture - BREAKOUT is claimed by several independently-sourced
    methodologies (Darvas, Turtle, CANSLIM, SEPA, ORB)."""
    conn = db.connect(":memory:")
    ingest_corpus(conn, RuleBasedMethodologyExtractor(), SEED_CORPUS, NOW)
    contributors = db.methodologies_for_concept(conn, "BREAKOUT")
    assert len({r["methodology_id"] for r in contributors}) >= 3


def test_seed_corpus_entries_never_state_a_profitability_claim():
    """Structural check on schema.py's own guarantee - RawMethodologySource
    has no performance-figure field, so this asserts none of the free-text
    descriptions smuggle one in as prose either."""
    banned_terms = ["win rate", "% return", "guaranteed", "always profit", "beats the market"]
    for source in SEED_CORPUS:
        lowered = source.raw_text.lower()
        for term in banned_terms:
            assert term not in lowered, f"{source.name} description contains a profitability claim: {term!r}"
