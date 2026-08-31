"""Items 5/6/8: turns a SourceDocument into structured TimelineEvent(s) -
event/claim extraction, evidence-type tagging (FACT/REPORTING/
INTERPRETATION/SPECULATION - item 8), and CLAUSE-LEVEL sentiment (item 6,
so "Revenue increased strongly, but margins will decline" becomes TWO
events - revenue: POSITIVE, margins: NEGATIVE - never one averaged-out
"mixed" number).

RULE-BASED, DISCLOSED, NOT LLM-BACKED IN THIS ENVIRONMENT BY DEFAULT: see
llm_synthesis.py for the explicit LLM_STATUS mechanism (item 19) - when no
LLM client is configured (the default in this environment, same
no-silent-fallback discipline as market_agent/llm/select.py), extraction
runs through fixed keyword/lexicon rules below. This is a genuine, tested,
useful v1 - not a placeholder - but it is coarser than true semantic
extraction would be, and every report says so explicitly via `llm_status`.

CLAUSE SPLITTING: a sentence is split on CONTRAST conjunctions ("but",
"however", "although", "while", "yet", "whereas") before sentiment/
evidence-type scoring, because those words are exactly where a single
sentence legitimately carries two different claims with two different
polarities - the user's own worked example ("Revenue increased strongly,
BUT management warned margins will decline").
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from market_agent.research.schema import TimelineEvent
from market_agent.research.providers import SourceDocument

CONTRAST_SPLIT_RE = re.compile(r"\b(but|however|although|while|yet|whereas|though)\b", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- evidence-type cue words (item 8), priority order: SPECULATION > INTERPRETATION > REPORTING ---
SPECULATION_CUES = ("could", "might", "may ", "likely to", "expected to", "analysts expect", "investors may",
                     "could see", "may see", "is expected", "are expected", "forecast to", "projected to")
INTERPRETATION_CUES = ("suggests", "reflects", "appears to", "appears that", "indicates", "signals",
                        "raises questions", "may indicate", "points to", "underscores")
REPORTING_CUES = ("said", "according to", "reported", "told", "announced", "stated", "disclosed",
                   "confirmed", "noted", "warned", "said in a statement", "wrote in a filing")

# --- sentiment lexicon - simple, disclosed, fixed polarity word lists (not ML) ---
POSITIVE_WORDS = ("increase", "increased", "increases", "growth", "grew", "grows", "strong", "strength",
                   "beat", "beats", "exceeded", "exceed", "record", "raised", "raises", "raising",
                   "surge", "surged", "rally", "rallied", "rallying", "outperform", "upgrade", "upgraded",
                   "expand", "expanded", "expansion", "profit", "profitable", "gain", "gains", "gained",
                   "improve", "improved", "improving", "robust", "accelerate", "accelerating", "bullish",
                   "optimistic", "buyback", "resilient", "boost", "boosted",
                   "climb", "climbed", "climbing", "rose", "rising", "jumped", "jumping", "soared", "soaring",
                   "advanced", "advancing",
                   # common financial-reporting phrasing not covered by the single words above
                   "above expectations", "beat expectations", "beats expectations", "topped estimates",
                   "topped forecasts", "ahead of forecasts", "ahead of estimates", "lifted its outlook",
                   "lifted outlook", "well above", "better than expected")
NEGATIVE_WORDS = ("decline", "declined", "declines", "declining", "fall", "fell", "falling", "falls",
                   "drop", "dropped", "drops", "weak", "weakness", "miss", "missed", "misses", "cut",
                   "cuts", "cutting", "lowered", "lowers", "lowering", "layoffs", "lawsuit", "investigation",
                   "probe", "recall", "shortfall", "concern", "concerns", "concerned", "worry", "worried",
                   "risk", "risks", "warn", "warned", "warns", "slowdown", "slowing", "slump", "slumped",
                   "bearish", "downgrade", "downgraded", "contraction", "contract", "loss", "losses", "delay",
                   "delayed", "resign", "resigned", "resignation", "controversy", "scrutiny", "fraud",
                   "default", "bankruptcy", "impairment",
                   "tumbled", "tumbling", "plunged", "plunging", "slid", "sliding", "sank", "sinking",
                   "pulling back", "pulled back", "pull back",
                   # common financial-reporting phrasing not covered by the single words above
                   "below expectations", "missed expectations", "misses expectations", "fell short",
                   "below forecasts", "below estimates", "worse than expected")

_POSITIVE_WORD_RE = [re.compile(rf"\b{re.escape(w)}\b") for w in POSITIVE_WORDS]
_NEGATIVE_WORD_RE = [re.compile(rf"\b{re.escape(w)}\b") for w in NEGATIVE_WORDS]

# NEGATION, as its own signal separate from polarity words - SENTiVENT (Jacobs & Hoste; see the
# annotation-guideline repo github.com/GillesJ/sentivent-event-annotation-guidelines) annotates
# negation as an explicit event attribute alongside type/subtype/modality, precisely because "revenue
# did not decline" and "revenue declined" must not collapse to the same polarity. Deliberately excludes
# bare "no" - "no doubt", "no surprise", "no signs of slowing" are common financial-reporting idioms
# that use "no" as an INTENSIFIER, not a negator, and including it produced exactly that false-positive
# class in practice. This is presence-based, clause-scoped negation, not real negation-scope parsing
# (no dependency parser is available in this environment) - a known, disclosed miss is an idiom like
# "not only strong but also improving", where "not" is part of a correlative construction rather than
# a true negator. See tests/test_extraction_negation.py for what is and isn't covered.
NEGATION_CUES = ("not", "never", "cannot", "fails to", "failed to", "unable to", "denies", "denied", "without")
_NEGATION_CUE_RE = [re.compile(rf"\b{re.escape(w)}\b") for w in NEGATION_CUES]


def _is_negated(lower_clause: str) -> bool:
    return "n't" in lower_clause or any(p.search(lower_clause) for p in _NEGATION_CUE_RE)

# --- affected-area keyword tags ---
AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "top line"),
    "margins": ("margin", "margins", "gross margin", "operating margin"),
    "guidance": ("guidance", "outlook", "forecast"),
    "earnings": ("earnings", "eps", "profit", "net income"),
    "cash_flow": ("cash flow", "free cash flow", "operating cash"),
    "debt": ("debt", "leverage", "credit rating"),
    "management": ("ceo", "cfo", "chief executive", "chief financial", "management", "executive"),
    "product": ("product", "launch", "unveil", "release"),
    "regulatory": ("regulatory", "regulator", "lawsuit", "antitrust", "investigation", "sec probe",
                    "fine", "sanction"),
    "supply_chain": ("supply chain", "chip shortage", "shortage", "supplier", "manufacturing"),
    "competition": ("competitor", "competition", "rival", "market share"),
    "workforce": ("layoff", "layoffs", "workforce", "headcount", "hiring"),
    "capital_allocation": ("buyback", "dividend", "repurchase", "capital return"),
    "geopolitical": ("china", "tariff", "export control", "sanctions", "geopolitical"),
    "demand": ("demand", "orders", "bookings", "backlog"),
}

# --- event-type classification, derived from affected_area + document source type ---
MATERIAL_AREA_TO_EVENT_TYPE = {
    "guidance": "GUIDANCE_CHANGE", "earnings": "EARNINGS", "capital_allocation": "DIVIDEND_CHANGE",
    "management": "MANAGEMENT_CHANGE", "regulatory": "REGULATORY", "workforce": "WORKFORCE_CHANGE",
    "product": "PRODUCT", "supply_chain": "SUPPLY_CHAIN", "geopolitical": "GEOPOLITICAL",
    "competition": "COMPETITION", "debt": "CAPITAL_STRUCTURE",
}
MATERIAL_EVENT_TYPES = {"GUIDANCE_CHANGE", "EARNINGS", "DIVIDEND_CHANGE", "MANAGEMENT_CHANGE", "REGULATORY",
                         "CAPITAL_STRUCTURE", "ACQUISITION_DIVESTITURE", "RESTRUCTURING", "MATERIAL_AGREEMENT",
                         "OWNERSHIP_CHANGE"}

MIN_CLAUSE_LENGTH = 25  # characters - shorter fragments are usually not standalone claims


def _split_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        parts = CONTRAST_SPLIT_RE.split(sentence)
        # re.split with a capturing group interleaves the matched conjunctions - drop them, keep the text parts
        for i, part in enumerate(parts):
            if i % 2 == 0:
                part = part.strip(" ,.;:")
                if part:
                    clauses.append(part)
    return clauses


def classify_evidence_type(clause: str, source_type: str) -> str:
    if source_type == "SEC_FILING":
        return "FACT"  # primary company disclosure - see module docstring's disclosed simplification
    lower = clause.lower()
    if any(cue in lower for cue in SPECULATION_CUES):
        return "SPECULATION"
    if any(cue in lower for cue in INTERPRETATION_CUES):
        return "INTERPRETATION"
    if any(cue in lower for cue in REPORTING_CUES):
        return "REPORTING"
    return "REPORTING"  # a plain declarative sentence in a NEWS article is still secondhand, not primary FACT


def classify_sentiment(clause: str) -> str:
    """Word-boundary matching, not plain substring containment - the same
    fix classify_affected_area already applies (see its own docstring for
    the "product" vs "production" case). Plain `in` containment let
    "executives" silently match the NEGATIVE cue "cut" (e-x-e-CUT-ives),
    flipping a genuinely positive clause about strong demand to MIXED -
    found via a real paraphrase-invariance validation run, not a
    contrived example.

    NEGATION FLIPS A ONE-SIDED POLARITY READ: if the clause matches only
    positive words (or only negative words) AND carries a negation cue
    (_is_negated, see above), the polarity is inverted - "revenue did not
    decline" must not read the same as "revenue declined". A MIXED clause
    (both polarities already present) is left as MIXED rather than guessing
    which side the negation scopes over - resolving that would need real
    negation-scope parsing this environment doesn't have."""
    lower = clause.lower()
    pos = sum(1 for w in _POSITIVE_WORD_RE if w.search(lower))
    neg = sum(1 for w in _NEGATIVE_WORD_RE if w.search(lower))
    negated = _is_negated(lower)
    if pos > 0 and neg == 0:
        return "NEGATIVE" if negated else "POSITIVE"
    if neg > 0 and pos == 0:
        return "POSITIVE" if negated else "NEGATIVE"
    if pos > 0 and neg > 0:
        return "MIXED"
    return "NEUTRAL"


def classify_affected_area(clause: str) -> str | None:
    """Word-boundary matching, not plain substring containment - "product"
    must not match inside "production" (found via a failing test: "supply
    chain shortage affected production" was misclassified as `product`
    purely because "product" is a substring of "production")."""
    lower = clause.lower()
    # multi-word keywords ("supply chain") are checked before single-word ones, and dict order is
    # otherwise preserved - a longer, more specific phrase should win over a shorter generic word.
    for area, keywords in AREA_KEYWORDS.items():
        for kw in sorted(keywords, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return area
    return None


# Real SEC 8-K item-code taxonomy (Regulation S-K Item 601/8-K form instructions), used to give a
# filing-index document (which carries no substantive filing TEXT in this MVP - see providers.py's
# module docstring on that disclosed scope limit) a meaningful event_type instead of dumping every
# filing into one undifferentiated "SEC_DISCLOSURE" bucket. Checked BEFORE the generic area-keyword
# fallback since an item code is a real, structured SEC classification, not a heuristic guess.
ITEM_CODE_TO_EVENT_TYPE: dict[str, str] = {
    "1.01": "MATERIAL_AGREEMENT", "1.02": "MATERIAL_AGREEMENT", "2.01": "ACQUISITION_DIVESTITURE",
    "2.02": "EARNINGS", "2.03": "CAPITAL_STRUCTURE", "2.04": "CAPITAL_STRUCTURE", "2.05": "RESTRUCTURING",
    "2.06": "RESTRUCTURING", "3.01": "REGULATORY", "4.01": "MANAGEMENT_CHANGE", "5.01": "OWNERSHIP_CHANGE",
    "5.02": "MANAGEMENT_CHANGE", "5.03": "CORPORATE_GOVERNANCE", "5.07": "CORPORATE_GOVERNANCE",
    "7.01": "DISCLOSURE", "8.01": "GENERAL_CORPORATE", "9.01": "GENERAL_CORPORATE",
}
ITEM_CODES_RE = re.compile(r"item codes?:\s*([0-9.,\s]+)", re.IGNORECASE)


def _filing_event_type_hint(document_content: str) -> str | None:
    """The FIRST recognized item code on an 8-K, in the order SEC lists
    them - a filing can carry several item codes but this MVP picks one
    representative event_type per filing rather than trying to attribute
    individual clauses to individual item codes (the filing-index document
    has no per-item text to split on in the first place - see module
    docstring's disclosed scope limit)."""
    match = ITEM_CODES_RE.search(document_content)
    if not match:
        return None
    for code in match.group(1).split(","):
        event_type = ITEM_CODE_TO_EVENT_TYPE.get(code.strip())
        if event_type:
            return event_type
    return None


def _event_type_for(area: str | None, source_type: str, filing_hint: str | None = None) -> str:
    if source_type == "SEC_FILING":
        return MATERIAL_AREA_TO_EVENT_TYPE.get(area) or filing_hint or "PERIODIC_FILING"
    return MATERIAL_AREA_TO_EVENT_TYPE.get(area, "GENERAL_NEWS")


def _confidence_for(evidence_type: str, reliability: str) -> str:
    if evidence_type == "FACT":
        return "HIGH"
    if evidence_type == "SPECULATION":
        return "LOW"
    return "HIGH" if reliability == "PRIMARY" else ("MEDIUM" if reliability == "SECONDARY" else "LOW")


def _materiality_for(event_type: str, source_type: str) -> str:
    if source_type == "SEC_FILING" or event_type in MATERIAL_EVENT_TYPES:
        return "HIGH"
    if event_type == "GENERAL_NEWS":
        return "LOW"
    return "MEDIUM"


def extract_events(document: SourceDocument) -> list[TimelineEvent]:
    """One TimelineEvent per clause with enough signal to stand alone -
    see module docstring. A document with no recognizable clauses (too
    short, no keyword match, pure boilerplate) produces zero events,
    never a fabricated one."""
    events: list[TimelineEvent] = []
    filing_hint = _filing_event_type_hint(document.raw_content) if document.source_type == "SEC_FILING" else None
    for i, clause in enumerate(_split_clauses(document.normalized_content)):
        if len(clause) < MIN_CLAUSE_LENGTH:
            continue
        area = classify_affected_area(clause)
        if area is None and document.source_type != "SEC_FILING":
            continue  # a news clause with no recognizable financial/business topic isn't a usable claim
        evidence_type = classify_evidence_type(clause, document.source_type)
        sentiment = classify_sentiment(clause)
        event_type = _event_type_for(area, document.source_type, filing_hint)
        event_id = hashlib.sha256(f"{document.source_id}:{i}".encode()).hexdigest()[:16]
        events.append(TimelineEvent(
            event_id=event_id, entity=document.entity, date=document.published_at[:10],
            event_type=event_type, description=clause, evidence_type=evidence_type,
            source_ids=[document.source_id], confidence=_confidence_for(evidence_type, document.reliability),
            materiality=_materiality_for(event_type, document.source_type), sentiment=sentiment,
            affected_area=area,
        ))
    return events


def extract_all_events(documents: list[SourceDocument]) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for doc in documents:
        if doc.duplicate_of is not None:
            continue  # syndicated copy - the canonical document already contributes its own events
        events.extend(extract_events(doc))
    return events
