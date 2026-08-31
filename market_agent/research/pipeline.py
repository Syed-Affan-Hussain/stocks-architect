"""Items 21/22: the pipeline orchestrator and the product's public API.
Implements exactly the pipeline the product spec lays out:

USER -> COMPANY IDENTIFICATION -> DATA COLLECTION -> SOURCE NORMALIZATION
-> DEDUPLICATION -> EVENT/CLAIM EXTRACTION -> NEWS SENTIMENT -> NARRATIVE
DETECTION -> QUANTIFIED NEWS STATE -> COMPANY FUNDAMENTALS -> MARKET
CONTEXT -> HISTORICAL EVENT ANALYSIS -> NARRATIVE vs FUNDAMENTALS ->
CONTRADICTION DETECTION -> RISK ANALYSIS -> CHANGE DETECTION -> AI
RESEARCH SYNTHESIS -> FINAL REPORT

QUANTIFIED NEWS STATE (market_agent/research/news_state/) is ADDITIVE: it
runs alongside narrative detection over the same documents, and its output
is surfaced on the report (ResearchReport.news_state) without feeding the
existing narrative/consistency/risk/assessment logic below it - see that
field's own docstring in schema.py for why that boundary is deliberate.

`research_company` is the ONLY function that runs the full pipeline; every
other public API function (get_company_profile, get_company_timeline, ...)
calls it and extracts one field - the CLI (research/__main__.py) is
required to go through these, never re-implement pipeline logic (item 22).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from market_agent.llm.interpreter import LLMClient
from market_agent.research.assessment import build_assessment, what_would_change_the_view
from market_agent.research.change_detection import detect_changes
from market_agent.research.consistency import check_consistency, detect_contradictions
from market_agent.research.fundamentals import build_fundamental_facts
from market_agent.research.historical_reaction import historical_reactions_for_recent_event_types, open_historical_ledger
from market_agent.research.extraction import extract_all_events
from market_agent.research.llm_synthesis import llm_status, synthesize_executive_summary
from market_agent.research.market_context import build_market_context
from market_agent.research.narratives import cluster_narratives
from market_agent.research.news_state.pipeline import build_news_state_from_documents
from market_agent.research.normalize import deduplicate_documents
from market_agent.research.providers import NewsProvider, SECProvider
from market_agent.research.risk import extract_all_risks
from market_agent.research.schema import SOURCE_UNAVAILABLE, ChangeSummary, CompanyProfile, ResearchReport
from market_agent.sources.yahoo_prices import YahooPriceSeriesProvider
from market_agent.store import db

DEFAULT_DB_PATH = "data_cache/research/market_agent_research.sqlite"
DEFAULT_LOOKBACK_DAYS = 30
FILING_LOOKBACK_MULTIPLIER = 6  # material SEC filings are sparser than news - look back further for them,
#                                  disclosed rather than silently missing an 8-K/10-Q just outside a 30-day window
HISTORY_EVENT_TYPES = ("GUIDANCE_CHANGE", "DIVIDEND_CHANGE")  # the ONLY event types the reused, existing
#                        historical ledger (stages 1-7) actually has real cross-company data for - see
#                        historical_reaction.py's module docstring

# Legal-suffix stripping so the SEC-registered name ("NVIDIA CORP") also yields the shorter form
# ("NVIDIA") real news prose actually uses - a best-effort string operation, not a real name-variant
# database, so uncommon suffixes or a name news prose shortens differently will simply not get a
# second alias (degrades to just the raw registered name, still better than none).
_COMPANY_NAME_SUFFIXES = (" CORPORATION", " INCORPORATED", " HOLDINGS", " COMPANY", " LIMITED",
                           " CORP", " INC", " LTD", " PLC", " LLC", " CO")


def _derive_name_aliases(raw_name: str | None) -> tuple[str, ...]:
    """entity_resolution.py's attribution gate needs to recognize the
    queried company by the name real news actually uses, not just its
    ticker - see this function's call site."""
    if not raw_name:
        return ()
    upper = raw_name.upper()
    for suffix in _COMPANY_NAME_SUFFIXES:
        if upper.endswith(suffix):
            stripped = raw_name[: -len(suffix)].strip()
            return (raw_name, stripped) if stripped else (raw_name,)
    return (raw_name,)


def research_company(ticker: str, conn: sqlite3.Connection | None = None,
                      prices: YahooPriceSeriesProvider | None = None, llm_client: LLMClient | None = None,
                      lookback_days: int = DEFAULT_LOOKBACK_DAYS, generated_at: datetime | None = None,
                      historical_ledger_conn: sqlite3.Connection | None = None) -> ResearchReport:
    """Item 18: running this again for the SAME entity against the SAME
    `conn` automatically becomes a continuous-research update - the prior
    report is read back from research_reports (schema v5) and diffed via
    change_detection.py, never re-derived from scratch."""
    entity = ticker.upper().strip()
    now = generated_at or datetime.now(timezone.utc)
    conn = conn or db.connect(DEFAULT_DB_PATH)
    prices = prices or YahooPriceSeriesProvider()
    ohlcv = prices  # YahooPriceSeriesProvider implements both PriceSeriesProvider and OHLCVProvider

    unavailable_sources: list[str] = []

    # --- DATA COLLECTION ---
    sec = SECProvider()
    filing_result = sec.fetch_recent_filings(entity, entity, lookback_days=lookback_days * FILING_LOOKBACK_MULTIPLIER)
    if filing_result.status == SOURCE_UNAVAILABLE:
        unavailable_sources.append("SEC_FILINGS")
    news_result = NewsProvider().fetch(entity, entity)
    if news_result.status == SOURCE_UNAVAILABLE:
        unavailable_sources.append("NEWS")
    facts_status, facts_json, _facts_evidence = sec.fetch_company_facts(entity)
    if facts_status == SOURCE_UNAVAILABLE:
        unavailable_sources.append("FUNDAMENTALS")

    all_documents = filing_result.documents + news_result.documents

    # --- SOURCE NORMALIZATION + DEDUPLICATION ---
    documents, _canonical_map = deduplicate_documents(all_documents)

    # --- EVENT/CLAIM EXTRACTION (+ NEWS SENTIMENT, clause-level, inside extraction.py) ---
    events = extract_all_events(documents)
    events_by_id = {e.event_id: e for e in events}

    # --- NARRATIVE DETECTION ---
    narratives = cluster_narratives(events, documents, now=now)

    # --- QUANTIFIED NEWS STATE (news_state/ - additive, does not feed assessment/consistency/risk
    # below; see ResearchReport.news_state's docstring). Reuses the SAME deduplicated `documents` this
    # pipeline already collected - its own internal dedup/extraction pass over them is idempotent, not
    # a second network round-trip. persist=True so `db.latest_news_company_state` (called inside)
    # picks up a real prior state on the NEXT run of the SAME entity against the SAME conn, giving
    # state_change/state_velocity/state_direction for free on continuous-research updates (item 18).
    # `meta` is fetched here (moved up from COMPANY FUNDAMENTALS below) so its registered company name
    # can be passed as an entity_resolution.py alias - real news prose almost never uses the bare
    # ticker, so without this the attribution gate's entity-mention check would rarely fire.
    meta = sec.fetch_company_meta(entity)
    entity_aliases = _derive_name_aliases(meta.get("name") if meta else None)
    news_company_state = None
    if news_result.status != SOURCE_UNAVAILABLE:
        news_company_state, _news_event_vectors = build_news_state_from_documents(
            entity, documents, conn, as_of=now, persist=True, entity_aliases=entity_aliases)

    # --- COMPANY FUNDAMENTALS ---
    fundamentals_source_id = f"sec-xbrl:{entity}" if facts_json else None
    fundamental_facts = build_fundamental_facts(facts_json, fundamentals_source_id)
    profile = CompanyProfile(
        entity=entity, cik=sec.resolve_cik(entity), name=meta.get("name") if meta else None,
        sic_description=meta.get("sicDescription") if meta else None, fundamentals=fundamental_facts,
        profile_source_ids=[d.source_id for d in documents if d.source_type == "SEC_FILING"],
        last_updated=now.isoformat(),
    )

    # --- MARKET CONTEXT ---
    market_ctx = build_market_context(ohlcv, prices, entity, as_of=now)
    if market_ctx is None:
        unavailable_sources.append("MARKET_DATA")

    # --- HISTORICAL EVENT ANALYSIS ---
    hist_conn = historical_ledger_conn if historical_ledger_conn is not None else open_historical_ledger()
    recent_event_types = sorted({
        (e.event_type, "positive" if e.sentiment == "POSITIVE" else "negative")
        for e in events if e.event_type in HISTORY_EVENT_TYPES and e.sentiment in ("POSITIVE", "NEGATIVE")
    })
    historical_reactions = (historical_reactions_for_recent_event_types(hist_conn, recent_event_types)
                             if hist_conn is not None else [])
    if hist_conn is None:
        unavailable_sources.append("HISTORICAL_LEDGER")

    # --- NARRATIVE vs FUNDAMENTALS + CONTRADICTION DETECTION ---
    consistency_checks = check_consistency(narratives, facts_json)
    contradictions = detect_contradictions(narratives, events_by_id)

    # --- RISK ANALYSIS ---
    risks = extract_all_risks(narratives, events_by_id, contradictions, fundamental_facts, fundamentals_source_id)

    # --- ASSESSMENT (needed before change detection can report an assessment delta) ---
    assessment, confidence, reasoning = build_assessment(narratives, risks, consistency_checks)
    strongest, weakest, uncertainty, strengthen, weaken = what_would_change_the_view(narratives, risks,
                                                                                       consistency_checks)

    # --- CHANGE DETECTION (item 11/18) ---
    prior_row = db.latest_research_report(conn, entity)
    prior_json = json.loads(prior_row["report_json"]) if prior_row is not None else None
    change = detect_changes(narratives, risks, assessment, prior_json)

    # --- AI RESEARCH SYNTHESIS (item 19 - LLM-optional, explicit status) ---
    executive_summary = synthesize_executive_summary(llm_client, entity, assessment, narratives, risks)

    positive_factors = [n.description for n in narratives if n.sentiment == "POSITIVE"]
    what_to_watch = ([r.description for r in risks if r.status == "EMERGING"][:5]
                     or ["No specific emerging items identified in this pass - monitor for new filings and "
                         "news coverage."])

    report = ResearchReport(
        entity=entity, generated_at=now.isoformat(), llm_status=llm_status(llm_client),
        research_period_days=lookback_days, assessment=assessment, assessment_confidence=confidence,
        assessment_reasoning=reasoning, executive_summary=executive_summary, profile=profile, timeline=events,
        narratives=narratives, consistency_checks=consistency_checks, contradictions=contradictions, risks=risks,
        positive_factors=positive_factors, market_context=market_ctx, historical_reactions=historical_reactions,
        change=change, strongest_evidence=strongest, weakest_evidence=weakest, major_uncertainty=uncertainty,
        what_would_strengthen=strengthen, what_would_weaken=weaken, what_to_watch=what_to_watch,
        sources=documents, unavailable_sources=unavailable_sources,
        news_state=news_company_state.to_dict() if news_company_state is not None else None,
    )

    db.save_research_report(conn, entity, now, assessment, report.to_dict())
    return report


# --- item 22: the public internal API - every function below is a thin wrapper over research_company,
# never a second, duplicated implementation ---

def get_company_profile(ticker: str, **kwargs) -> CompanyProfile:
    return research_company(ticker, **kwargs).profile


def get_company_timeline(ticker: str, **kwargs) -> list:
    return research_company(ticker, **kwargs).timeline


def get_company_narratives(ticker: str, **kwargs) -> list:
    return research_company(ticker, **kwargs).narratives


def get_company_sentiment(ticker: str, **kwargs) -> dict:
    report = research_company(ticker, **kwargs)
    return {"positive": [n.description for n in report.narratives if n.sentiment == "POSITIVE"],
            "negative": [n.description for n in report.narratives if n.sentiment == "NEGATIVE"],
            "mixed": [n.description for n in report.narratives if n.sentiment == "MIXED"]}


def get_company_risks(ticker: str, **kwargs) -> list:
    return research_company(ticker, **kwargs).risks


def get_company_assessment(ticker: str, **kwargs) -> tuple[str, float | None, str]:
    report = research_company(ticker, **kwargs)
    return report.assessment, report.assessment_confidence, report.assessment_reasoning


def get_company_news_state(ticker: str, **kwargs) -> dict | None:
    return research_company(ticker, **kwargs).news_state
