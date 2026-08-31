"""The validation set required directly: real-style financial-news
examples testing the five properties the event quantifier must get right.
Every example is realistic financial-reporting phrasing (not lorem ipsum,
not the trivial one-line fixtures used elsewhere in this test suite) run
through the ACTUAL pipeline (build_news_state_from_documents /
cluster_events / build_event_vector) end to end - not isolated unit
assertions on hand-picked internals.

FIVE CATEGORIES, EACH ITS OWN TEST(S):
  1. Same event, different sources    -> must MERGE into one EventVector
  2. Different events, same topic     -> must NOT merge
  3. Different magnitudes             -> must score differently, and a
                                          large enough gap must SPLIT
                                          identity even within one window
  4. Paraphrases                      -> must produce the SAME structured
                                          implications despite different
                                          wording
  5. Contradictory evidence           -> must merge (same topic/window,
                                          no magnitude to split on) but
                                          show real dispersion + a
                                          contradiction flag, not a
                                          falsely confident average
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_agent.research.news_state.pipeline import build_news_state_from_documents
from market_agent.research.providers import make_fingerprint
from market_agent.research.schema import SourceDocument
from market_agent.store import db

NOW = datetime(2024, 8, 15, tzinfo=timezone.utc)


def _doc(source_id, title, content, publisher, reliability="TERTIARY", days_ago=0.0, entity="NVDA"):
    date = (NOW - timedelta(days=days_ago)).isoformat()
    return SourceDocument(source_id=source_id, publisher=publisher, source_type="NEWS", url=f"https://x/{source_id}",
                           published_at=date, retrieved_at=date, entity=entity, title=title, raw_content=content,
                           normalized_content=content, reliability=reliability,
                           fingerprint=make_fingerprint(title, content))


# --- 1. Same event, different sources (real-style paraphrases of one real earnings report) ---

def test_same_event_across_three_independent_sources_merges_into_one():
    docs = [
        _doc("s1", "NVIDIA quarterly revenue jumps on AI chip demand",
             "NVIDIA reported quarterly revenue grew 34% year over year, driven by continued strong demand for "
             "its AI data-center chips.", publisher="Reuters", reliability="SECONDARY", days_ago=2),
        _doc("s2", "NVIDIA posts 34% revenue growth on data-center strength",
             "NVIDIA's data-center business helped drive 34% year-over-year revenue growth in the latest "
             "quarter, the company said.", publisher="Bloomberg", reliability="SECONDARY", days_ago=1),
        _doc("s3", "Chipmaker NVIDIA beats on strong AI demand",
             "The chipmaker posted revenue up 34% from a year earlier, fueled by robust demand for AI "
             "processors.", publisher="CNBC", days_ago=0),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False)
    assert len(event_vectors) == 1
    ev = event_vectors[0]
    assert ev.implications["growth"] == 1.0  # 34% saturates the percent anchor - consistent across all 3 sources
    assert ev.independent_source_count == 3
    assert state.confidence > 0


# --- 2. Different events, same topic (two genuinely separate regulatory stories, weeks apart) ---

def test_different_events_same_topic_do_not_merge():
    docs = [
        _doc("r1", "NVIDIA faces EU antitrust inquiry over GPU licensing",
             "Regulators opened an antitrust investigation into NVIDIA's GPU licensing practices in Europe.",
             publisher="Reuters", reliability="SECONDARY", days_ago=40),
        _doc("r2", "NVIDIA settles unrelated US export-control probe",
             "NVIDIA reached a settlement with US authorities over a separate export-control investigation "
             "into chip sales.", publisher="Bloomberg", reliability="SECONDARY", days_ago=2),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False)
    assert len(event_vectors) == 2  # same topic (regulatory), 38 days apart - genuinely different occurrences


# --- 3. Different magnitudes (a modest quarter vs. an exceptional one) ---

def test_different_magnitudes_score_differently_and_split_identity_within_the_window():
    docs = [
        _doc("m1", "NVIDIA reports modest revenue growth",
             "NVIDIA reported revenue grew 3% year over year in the quarter.", publisher="Reuters",
             reliability="SECONDARY", days_ago=1),
        _doc("m2", "NVIDIA reports blockbuster revenue growth",
             "NVIDIA reported revenue grew 45% year over year in the quarter.", publisher="Bloomberg",
             reliability="SECONDARY", days_ago=0),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False)
    # same topic, 1 day apart (well within the window) - but a 3% vs 45% growth figure diverge by far
    # more than the 50% relative tolerance, so these are correctly treated as reporting DIFFERENT facts
    # (most likely two different quarters/metrics, not the same figure worded two ways).
    assert len(event_vectors) == 2
    scores = sorted(ev.implications["growth"] for ev in event_vectors)
    assert scores[0] < 0.3   # the 3% read
    assert scores[1] == 1.0  # the 45% read saturates
    assert scores[0] != scores[1]


# --- 4. Paraphrases (same real fact, worded very differently) ---

def test_paraphrases_of_the_same_fact_produce_identical_structured_implications():
    variants = [
        ("p1", "NVIDIA data-center revenue tops forecasts",
         "NVIDIA's data-center segment posted revenue growth of 30% year over year, well above analyst "
         "forecasts."),
        ("p2", "Data-center strength lifts NVIDIA results",
         "Strength in NVIDIA's data-center unit drove 30% year-on-year revenue growth, exceeding what "
         "Wall Street had penciled in."),
        ("p3", "NVIDIA's AI chip business keeps growing",
         "Sales in NVIDIA's AI-focused data-center business climbed 30% from the prior year."),
    ]
    conn = db.connect(":memory:")
    implications_seen = []
    for source_id, title, content in variants:
        doc = _doc(source_id, title, content, publisher=source_id, days_ago=0)
        state, event_vectors = build_news_state_from_documents("NVDA", [doc], conn, as_of=NOW, persist=False)
        implications_seen.append({k: v for k, v in event_vectors[0].implications.items() if v is not None})
    assert implications_seen[0] == implications_seen[1] == implications_seen[2]
    assert implications_seen[0]["growth"] == 1.0  # 30% saturates, identically, regardless of wording


# --- 5. Contradictory evidence (same topic/window, genuinely opposed claims) ---

def test_contradictory_demand_reports_merge_but_show_real_dispersion():
    docs = [
        _doc("c1", "NVIDIA says AI chip demand remains robust",
             "NVIDIA executives said demand for its AI chips remains strong heading into next quarter.",
             publisher="Reuters", reliability="SECONDARY", days_ago=1),
        _doc("c2", "Customers reportedly pulling back on AI chip orders",
             "Several customers are reportedly pulling back on orders amid concerns that AI chip demand is "
             "cooling.", publisher="The Information", days_ago=0),
    ]
    conn = db.connect(":memory:")
    state, event_vectors = build_news_state_from_documents("NVDA", docs, conn, as_of=NOW, persist=False)
    # same topic (demand), 1 day apart, no extractable magnitude on either side to split on - correctly
    # merges into ONE event rather than being treated as two unrelated stories.
    assert len(event_vectors) == 1
    ev = event_vectors[0]
    assert ev.implications["demand"] == 0.0        # moved to neutral - a real, opposed disagreement
    assert ev.dispersion["demand"] > 0               # NOT silently confident - the disagreement is visible
    # and it survives all the way to the aggregated company state, not just the single event:
    assert "demand" in state.contradiction_axes
