"""Core data structures for the AI Market Research & Analysis product
(market_agent/research/). This is a DELIBERATE PRODUCT PIVOT: everything
under events/, learn/, agents/, concepts/, outcomes/, reporting/,
setups/, strategy/ remains exactly as it was (the adaptive/statistical
trading-research system) and is REUSED here as supporting infrastructure,
never modified to serve this new purpose. This module defines the NEW
vocabulary the research product speaks: source documents (not
episodic_events), timeline events (not EventRecord/PredictionRecord),
narratives, risks, contradictions, and a final research report.

CORE DISCIPLINE, CARRIED OVER FROM THE REST OF THIS PROJECT: never fabricate
missing data. A field that cannot be sourced from a real provider stays
None/empty and is disclosed, never guessed at (see providers.py's
SOURCE_UNAVAILABLE convention). Every factual claim in a TimelineEvent or
FundamentalFact carries `source_ids` pointing back to the SourceDocument(s)
it came from - see report.py's citation rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# --- source reliability tiers (item 2) ---
SOURCE_RELIABILITY_LEVELS = ("PRIMARY", "SECONDARY", "TERTIARY")

# --- evidence type tiers (item 8) - never merged into one "factual statement" ---
EVIDENCE_TYPES = ("FACT", "REPORTING", "INTERPRETATION", "SPECULATION")

# --- narrative trend states (item 7) ---
NARRATIVE_TRENDS = ("EMERGING", "STRENGTHENING", "STABLE", "WEAKENING", "FADING", "DISPUTED")

# --- narrative-vs-fundamentals consistency (item 10) ---
CONSISTENCY_VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")

# --- risk classification (item 15) ---
RISK_CATEGORIES = ("BUSINESS", "VALUATION", "BALANCE_SHEET", "LIQUIDITY", "REGULATORY", "GEOPOLITICAL",
                    "COMPETITION", "CUSTOMER_CONCENTRATION", "SUPPLY_CHAIN", "EXECUTION", "NARRATIVE", "MARKET")
RISK_STATUSES = ("KNOWN", "EMERGING", "SPECULATIVE")

# --- overall research assessment (item 16) - deliberately NOT buy/sell ---
ASSESSMENTS = ("FAVORABLE", "CAUTIOUSLY_FAVORABLE", "NEUTRAL", "UNCERTAIN", "CAUTIOUS", "NEGATIVE",
               "INSUFFICIENT_EVIDENCE")

SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"  # item 1/28 - explicit, never silently substituted


@dataclass
class SourceDocument:
    """Item 2: every collected document, normalized to one structured
    shape regardless of which provider produced it."""
    source_id: str
    publisher: str
    source_type: str            # "SEC_FILING" | "NEWS" | "COMPANY_IR" | "MARKET_DATA"
    url: str | None
    published_at: str            # ISO
    retrieved_at: str             # ISO
    entity: str
    title: str
    raw_content: str
    normalized_content: str
    reliability: str               # one of SOURCE_RELIABILITY_LEVELS
    fingerprint: str                 # item 3 - near-duplicate detection key
    duplicate_of: str | None = None    # source_id of the canonical doc this is a syndication of, if any

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "publisher": self.publisher, "source_type": self.source_type,
                "url": self.url, "published_at": self.published_at, "retrieved_at": self.retrieved_at,
                "entity": self.entity, "title": self.title, "reliability": self.reliability,
                "fingerprint": self.fingerprint, "duplicate_of": self.duplicate_of}


@dataclass
class TimelineEvent:
    """Item 5: one material, dated event in a company's persistent
    history - the research product's analogue of a trading-system
    EventRecord, but never fed into episodic_events (different purpose,
    different schema, kept fully separate)."""
    event_id: str
    entity: str
    date: str                    # ISO
    event_type: str               # e.g. "GUIDANCE_CHANGE", "EARNINGS", "DIVIDEND_CHANGE", "MANAGEMENT_CHANGE", ...
    description: str
    evidence_type: str             # one of EVIDENCE_TYPES
    source_ids: list[str]
    confidence: str                  # "HIGH" | "MEDIUM" | "LOW"
    materiality: str                   # "HIGH" | "MEDIUM" | "LOW"
    sentiment: str                       # "POSITIVE" | "NEGATIVE" | "MIXED" | "NEUTRAL"
    affected_area: str | None = None       # e.g. "revenue", "margins", "guidance", "management"

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "entity": self.entity, "date": self.date,
                "event_type": self.event_type, "description": self.description,
                "evidence_type": self.evidence_type, "source_ids": self.source_ids,
                "confidence": self.confidence, "materiality": self.materiality, "sentiment": self.sentiment,
                "affected_area": self.affected_area}


@dataclass
class Narrative:
    """Item 7: a cluster of related timeline events/source documents
    telling ONE underlying story."""
    narrative_id: str
    entity: str
    description: str
    affected_area: str | None = None
    supporting_event_ids: list[str] = field(default_factory=list)
    contradicting_event_ids: list[str] = field(default_factory=list)
    first_observed: str | None = None
    latest_update: str | None = None
    source_count: int = 0
    independent_source_count: int = 0    # item 3 - after dedup, distinct underlying reports
    source_quality: str = "MIXED"          # dominant reliability tier among contributing sources
    sentiment: str = "NEUTRAL"
    confidence: str = "LOW"
    trend: str = "EMERGING"                  # one of NARRATIVE_TRENDS

    def to_dict(self) -> dict:
        return {"narrative_id": self.narrative_id, "entity": self.entity, "description": self.description,
                "affected_area": self.affected_area, "supporting_event_ids": self.supporting_event_ids,
                "contradicting_event_ids": self.contradicting_event_ids, "first_observed": self.first_observed,
                "latest_update": self.latest_update, "source_count": self.source_count,
                "independent_source_count": self.independent_source_count, "source_quality": self.source_quality,
                "sentiment": self.sentiment, "confidence": self.confidence, "trend": self.trend}


@dataclass
class FundamentalFact:
    """Item 4/9: one disclosed financial fact, always with provenance -
    never fabricated. `value=None` with a populated `source_ids=[]` means
    genuinely unavailable (SOURCE_UNAVAILABLE), not zero."""
    tag: str              # e.g. "Revenues", "NetIncomeLoss", "FreeCashFlow" (derived)
    label: str
    period_end: str | None
    value: float | None
    unit: str
    fiscal_period_type: str    # "FY" | "Q"
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tag": self.tag, "label": self.label, "period_end": self.period_end, "value": self.value,
                "unit": self.unit, "fiscal_period_type": self.fiscal_period_type, "source_ids": self.source_ids}


@dataclass
class CompanyProfile:
    """Item 4: persistent company research profile. Every field is
    Optional/empty when genuinely unavailable - never guessed."""
    entity: str
    cik: str | None
    name: str | None
    description: str | None = None
    sic_description: str | None = None      # closest available proxy for "industry/sector" (item 4) - SEC's own
    #                                          SIC classification; a real, disclosed source, not a fabricated one
    fundamentals: list[FundamentalFact] = field(default_factory=list)
    profile_source_ids: list[str] = field(default_factory=list)
    last_updated: str | None = None

    def to_dict(self) -> dict:
        return {"entity": self.entity, "cik": self.cik, "name": self.name, "description": self.description,
                "sic_description": self.sic_description, "fundamentals": [f.to_dict() for f in self.fundamentals],
                "profile_source_ids": self.profile_source_ids, "last_updated": self.last_updated}


@dataclass
class ConsistencyCheck:
    """Item 10: does the narrative match the disclosed fundamentals?"""
    narrative_id: str
    verdict: str            # one of CONSISTENCY_VERDICTS
    explanation: str
    supporting_fact_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"narrative_id": self.narrative_id, "verdict": self.verdict, "explanation": self.explanation,
                "supporting_fact_tags": self.supporting_fact_tags}


@dataclass
class Contradiction:
    """Item 14: an explicit conflict between two pieces of evidence."""
    contradiction_id: str
    description: str
    side_a: str
    side_a_source_ids: list[str]
    side_b: str
    side_b_source_ids: list[str]
    what_would_resolve_it: str

    def to_dict(self) -> dict:
        return {"contradiction_id": self.contradiction_id, "description": self.description, "side_a": self.side_a,
                "side_a_source_ids": self.side_a_source_ids, "side_b": self.side_b,
                "side_b_source_ids": self.side_b_source_ids, "what_would_resolve_it": self.what_would_resolve_it}


@dataclass
class Risk:
    """Item 15."""
    risk_id: str
    category: str          # one of RISK_CATEGORIES
    status: str              # one of RISK_STATUSES
    description: str
    severity: str               # "HIGH" | "MEDIUM" | "LOW"
    confidence: str                # "HIGH" | "MEDIUM" | "LOW"
    evidence_source_ids: list[str] = field(default_factory=list)
    recent_change: str | None = None
    affected_area: str | None = None

    def to_dict(self) -> dict:
        return {"risk_id": self.risk_id, "category": self.category, "status": self.status,
                "description": self.description, "severity": self.severity, "confidence": self.confidence,
                "evidence_source_ids": self.evidence_source_ids, "recent_change": self.recent_change,
                "affected_area": self.affected_area}


@dataclass
class HistoricalReaction:
    """Item 13: DESCRIPTIVE historical association, never framed as
    prediction - see historical_reaction.py's module docstring."""
    event_type: str
    direction: str
    horizon_days: int
    n: int
    median_reaction: float | None
    pct_positive: float | None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"event_type": self.event_type, "direction": self.direction, "horizon_days": self.horizon_days,
                "n": self.n, "median_reaction": self.median_reaction, "pct_positive": self.pct_positive,
                "evidence": self.evidence}


@dataclass
class MarketContext:
    """Item 12: descriptive price/technical context, never a
    prediction."""
    as_of: str
    price: float | None
    return_1m: float | None
    return_3m: float | None
    trend_direction: str
    volatility_state: str
    regime: str
    narrative_text: str

    def to_dict(self) -> dict:
        return {"as_of": self.as_of, "price": self.price, "return_1m": self.return_1m,
                "return_3m": self.return_3m, "trend_direction": self.trend_direction,
                "volatility_state": self.volatility_state, "regime": self.regime,
                "narrative_text": self.narrative_text}


@dataclass
class ChangeSummary:
    """Item 11/18: what changed since the last research pass for this
    entity."""
    has_prior_report: bool
    new_events: list[str] = field(default_factory=list)
    sentiment_change: str | None = None
    assessment_change: str | None = None
    new_risks: list[str] = field(default_factory=list)
    narrative_changes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"has_prior_report": self.has_prior_report, "new_events": self.new_events,
                "sentiment_change": self.sentiment_change, "assessment_change": self.assessment_change,
                "new_risks": self.new_risks, "narrative_changes": self.narrative_changes,
                "evidence": self.evidence}


@dataclass
class ResearchReport:
    """Item 16/21/22: the final, complete company research report -
    fully structured (JSON-ready via to_dict) and rendered as text by
    report_format.py."""
    entity: str
    generated_at: str
    llm_status: str                  # "UNAVAILABLE" | "ACTIVE:<model>" - item 19, never silent
    research_period_days: int
    assessment: str                    # one of ASSESSMENTS
    assessment_confidence: float | None   # 0-1
    assessment_reasoning: str
    executive_summary: str
    profile: CompanyProfile
    timeline: list[TimelineEvent]
    narratives: list[Narrative]
    consistency_checks: list[ConsistencyCheck]
    contradictions: list[Contradiction]
    risks: list[Risk]
    positive_factors: list[str]
    market_context: MarketContext | None
    historical_reactions: list[HistoricalReaction]
    change: ChangeSummary
    strongest_evidence: str
    weakest_evidence: str
    major_uncertainty: str
    what_would_strengthen: str
    what_would_weaken: str
    what_to_watch: list[str]
    sources: list[SourceDocument]
    unavailable_sources: list[str] = field(default_factory=list)   # item 1/28 - disclosed gaps
    news_state: dict | None = None   # news_state.schema.CompanyNewsState.to_dict() - the quantified,
    #   magnitude-aware, multi-axis news representation (market_agent/research/news_state/), computed
    #   over the SAME deduplicated documents as `narratives`/`timeline` above. Deliberately ADDITIVE:
    #   this does not feed assessment.py/consistency.py/risk.py's existing qualitative logic - it is
    #   surfaced alongside it, not wired into the decision engine. None only if news itself was
    #   SOURCE_UNAVAILABLE (see `unavailable_sources`); with zero real events it is still a real,
    #   mostly-null CompanyNewsState, not None - the engine ran, it just had nothing to report.

    def to_dict(self) -> dict:
        return {
            "entity": self.entity, "generated_at": self.generated_at, "llm_status": self.llm_status,
            "research_period_days": self.research_period_days, "assessment": self.assessment,
            "assessment_confidence": self.assessment_confidence, "assessment_reasoning": self.assessment_reasoning,
            "executive_summary": self.executive_summary, "profile": self.profile.to_dict(),
            "timeline": [e.to_dict() for e in self.timeline], "narratives": [n.to_dict() for n in self.narratives],
            "consistency_checks": [c.to_dict() for c in self.consistency_checks],
            "contradictions": [c.to_dict() for c in self.contradictions], "risks": [r.to_dict() for r in self.risks],
            "positive_factors": self.positive_factors,
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "historical_reactions": [h.to_dict() for h in self.historical_reactions],
            "change": self.change.to_dict(), "strongest_evidence": self.strongest_evidence,
            "weakest_evidence": self.weakest_evidence, "major_uncertainty": self.major_uncertainty,
            "what_would_strengthen": self.what_would_strengthen, "what_would_weaken": self.what_would_weaken,
            "what_to_watch": self.what_to_watch, "sources": [s.to_dict() for s in self.sources],
            "unavailable_sources": self.unavailable_sources, "news_state": self.news_state,
        }
