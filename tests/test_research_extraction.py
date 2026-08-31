from market_agent.research.extraction import (
    classify_affected_area, classify_evidence_type, classify_sentiment, extract_all_events, extract_events,
)
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument


def _doc(content, source_type="NEWS", reliability="TERTIARY", title="Test article"):
    return SourceDocument(source_id="d1", publisher="Pub", source_type=source_type, url="https://x",
                           published_at="2024-06-01T00:00:00+00:00", retrieved_at="2024-06-01T00:00:00+00:00",
                           entity="ACME", title=title, raw_content=content, normalized_content=content,
                           reliability=reliability, fingerprint=make_fingerprint(title, content))


def test_mixed_clause_splits_into_two_events_with_opposite_sentiment():
    doc = _doc("Revenue increased strongly, but management warned that margins will decline.")
    events = extract_events(doc)
    assert len(events) == 2
    revenue_event = next(e for e in events if e.affected_area == "revenue")
    margin_event = next(e for e in events if e.affected_area == "margins")
    assert revenue_event.sentiment == "POSITIVE"
    assert margin_event.sentiment == "NEGATIVE"


def test_sec_filing_clause_is_classified_as_fact_with_high_confidence():
    doc = _doc("The company reported revenue of $30.0 billion for the quarter.", source_type="SEC_FILING",
                reliability="PRIMARY")
    events = extract_events(doc)
    assert events
    assert events[0].evidence_type == "FACT"
    assert events[0].confidence == "HIGH"
    assert events[0].materiality == "HIGH"


def test_reporting_vs_speculation_vs_interpretation_cues():
    assert classify_evidence_type("The company said revenue grew.", "NEWS") == "REPORTING"
    assert classify_evidence_type("Analysts expect margins could decline next quarter.", "NEWS") == "SPECULATION"
    assert classify_evidence_type("The drop in orders suggests weakening demand.", "NEWS") == "INTERPRETATION"


def test_sentiment_classification():
    assert classify_sentiment("Revenue grew strongly and margins improved.") == "POSITIVE"
    assert classify_sentiment("Sales declined and layoffs were announced.") == "NEGATIVE"
    assert classify_sentiment("Revenue grew but margins declined.") == "MIXED"
    assert classify_sentiment("The company held its annual meeting today.") == "NEUTRAL"


def test_affected_area_classification():
    assert classify_affected_area("Revenue guidance was raised for the full year.") in ("revenue", "guidance")
    assert classify_affected_area("The CEO announced her resignation.") == "management"
    assert classify_affected_area("A supply chain shortage affected production.") == "supply_chain"
    assert classify_affected_area("Nothing recognizable here at all.") is None


def test_short_and_topicless_news_clauses_produce_no_events():
    doc = _doc("OK. Fine. Yes.")
    assert extract_events(doc) == []


def test_extract_all_events_skips_duplicate_documents():
    canonical = _doc("Revenue increased strongly this quarter for the company.")
    canonical.source_id = "canonical"
    duplicate = _doc("Revenue increased strongly this quarter for the company.")
    duplicate.source_id = "dupe"
    duplicate.duplicate_of = "canonical"
    events = extract_all_events([canonical, duplicate])
    assert all(e.source_ids == ["canonical"] for e in events)
