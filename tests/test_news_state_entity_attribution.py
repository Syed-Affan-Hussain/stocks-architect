"""End-to-end adversarial attribution tests: build_news_state_from_documents
on a MIXED batch of real-style articles - some genuinely about the queried
company, some about a competitor, some industry-wide, some about
customers/suppliers - and verify the resulting CompanyNewsState only
reflects the company's own events, with the rest visibly excluded
(excluded_by_role), not silently dropped or wrongly blended in.

This is the adversarial complement to test_news_state_entity_resolution.py
(which tests resolve_entity_role in isolation): here the check is that the
attribution gate actually changes what the FULL pipeline produces, end to
end, through clustering and aggregation - not just what the classifier
function itself returns."""
from datetime import datetime, timedelta, timezone

from market_agent.research.news_state.pipeline import build_news_state_from_documents
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument
from market_agent.store import db

NOW = datetime(2024, 8, 15, tzinfo=timezone.utc)


def _doc(source_id, title, content, days_ago=0.0, publisher="Reuters", reliability="SECONDARY"):
    date = (NOW - timedelta(days=days_ago)).isoformat()
    return SourceDocument(source_id=source_id, publisher=publisher, source_type="NEWS", url=f"https://x/{source_id}",
                           published_at=date, retrieved_at=date, entity="NVDA", title=title, raw_content=content,
                           normalized_content=content, reliability=reliability,
                           fingerprint=make_fingerprint(title, content))


def test_competitor_growth_never_reaches_the_companys_own_growth_axis():
    """Adversarial: a competitor's strong revenue growth must not inflate
    the queried company's own growth axis, even though both clauses match
    the identical "revenue" area keyword."""
    docs = [
        _doc("s1", "Rival AMD posts strong quarter",
             "Rival AMD posted revenue growth of 40% year over year in its latest quarter."),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False,
                                                             entity_aliases=("NVIDIA",))
    assert len(event_vectors) == 0
    assert state.dimensions["growth"] is None
    assert state.excluded_by_role.get("COMPETITOR") == 1


def test_industry_wide_news_with_no_company_named_is_excluded():
    docs = [
        _doc("s1", "Chip sector faces new sanctions",
             "New sanctions are expected to hurt the broader industry over the coming quarter."),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False,
                                                             entity_aliases=("NVIDIA",))
    assert len(event_vectors) == 0
    assert state.excluded_by_role.get("INDUSTRY") == 1


def test_mixed_batch_keeps_only_the_companys_own_events():
    """The core adversarial scenario: one real company event, one
    competitor event, and one customer-side (still-attributable) event in
    the same batch. Only two of the three should end up shaping the
    state, and the exclusion must be visible, not silent."""
    docs = [
        _doc("s1", "NVIDIA posts strong data-center growth",
             "NVIDIA reported data-center revenue grew 30% year over year.", publisher="Reuters"),
        _doc("s2", "Rival AMD also reports growth",
             "Rival AMD reported revenue grew 40% year over year in the same quarter.", publisher="Bloomberg"),
        _doc("s3", "NVIDIA customers pull back",
             "Several customers are reportedly pulling back on orders amid softer demand.", publisher="CNBC"),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False,
                                                             entity_aliases=("NVIDIA",))
    # only the NVIDIA growth event and the customer-demand event contributed - AMD's did not
    assert len(event_vectors) == 2
    assert state.dimensions["growth"] == 1.0          # from NVIDIA's own 30% figure, saturating
    assert state.dimensions["demand"] is not None      # customer-side demand news still counted
    assert state.excluded_by_role == {"COMPETITOR": 1}
    # sanity: the excluded competitor clause's magnitude (40%, larger than NVIDIA's own 30%) never
    # leaked into growth - if it had, growth would have been an average pulled toward/above 1.0 from
    # two contributing clauses instead of exactly the single NVIDIA clause's own saturated value.


def test_subsidiary_and_counterparty_events_both_still_count():
    docs = [
        _doc("s1", "NVIDIA's newly acquired unit reports early strength",
             "The company's newly acquired subsidiary reported strong early sales this quarter."),
        _doc("s2", "Suppliers flag delays",
             "A key supplier warned of delays that could affect upcoming shipments.", publisher="Bloomberg"),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False,
                                                             entity_aliases=("NVIDIA",))
    assert len(event_vectors) == 2
    assert state.excluded_by_role == {}


def test_industry_claim_naming_the_company_is_kept_not_excluded():
    docs = [
        _doc("s1", "NVIDIA warns on new sanctions",
             "NVIDIA warned that new sanctions could hurt the broader industry this year."),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False,
                                                             entity_aliases=("NVIDIA",))
    assert len(event_vectors) == 1
    assert state.excluded_by_role == {}


def test_without_alias_real_company_name_still_falls_back_correctly():
    """Even with NO entity_aliases supplied (the raw ticker "NVDA" never
    appears in real prose), ordinary self-attributed news must still work
    - the attribution gate defaults to SELF absent a third-party cue, so
    this doesn't depend on alias matching at all."""
    docs = [_doc("s1", "NVIDIA posts growth", "NVIDIA reported revenue grew 30% year over year.")]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False)
    assert len(event_vectors) == 1
    assert state.dimensions["growth"] == 1.0
