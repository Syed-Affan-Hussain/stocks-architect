"""Canonical entity + argument resolution for the news-state path -
subject -> action -> object -> affected-entity, so an event about a
DIFFERENT real-world party never gets counted as an event about the
queried company just because it shares a topic/keyword.

WHY THIS EXISTS: event_vector.py's IMPLICATION_RULES fire on AREA
keywords ("demand", "revenue", "workforce", ...) matched anywhere in a
clause, with no check on WHO the clause's subject actually is. "Rival AMD
posted 25% revenue growth" matches the same "revenue" area as "NVIDIA
posted 25% revenue growth" - before this module, both would have scored
identically as an NVIDIA growth event. This was a disclosed, deferred gap
in the News State Engine report (see the Structured Event Representation
literature card there) - this module closes it.

LITERATURE GROUNDING (both already verified in this project's earlier
review - see the News State Engine artifact's §8):
  - Han, Li, Qiao & Zheng, "Structured Event Representation and Stock
    Return Predictability" (arXiv:2512.19484): extracts events as
    (subject, action, object, context) triplets and standardizes
    subject/object to a CANONICAL identity so different mentions of the
    same real-world entity merge, and - implicitly - so different
    entities never merge. This module borrows the TRIPLE structure
    (subject_role/object_area here) and the CANONICALIZATION GOAL, not
    their mechanism: they resolve to DBpedia URLs via an LLM; this
    environment has no LLM client configured and no entity-linking
    service, so resolution here is deliberately coarser - a closed set
    of ROLES (SELF/COUNTERPARTY/COMPETITOR/SUBSIDIARY/INDUSTRY), not
    resolved real-world identities.
  - SENTiVENT (Jacobs & Hoste): annotates event PARTICIPANT ARGUMENTS as
    a first-class part of its schema, separate from the event trigger
    itself - the general principle that "who is this event about" is a
    distinct question from "what happened" is the reason subject_role is
    tracked as its own field here, not folded into affected_area.

WHAT THIS IS NOT: real named-entity recognition or coreference
resolution. There is no dependency parser or NER model in this
environment (same constraint disclosed throughout this package). Roles
are assigned from a small, disclosed set of keyword/pattern cues - see
each cue tuple's own comment for its known false-positive/false-negative
shape. `_extract_named_party` is a best-effort regex over capitalized
words, not a real entity extractor.

SCORING IS NOT TOUCHED: this module only decides whether a clause is
ELIGIBLE to contribute to the queried entity's implications at all
(`EventArguments.attributable`) - it assigns no score, weight, or
anchor. Every number a clause produces once it passes this gate is
computed exactly as it was in Event Quantifier v1.0 (event_vector.py,
magnitude.py, aggregation.py - unchanged by this module).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A rival/competitor cue followed by a capitalized name almost always means the clause is reporting
# on THAT NAMED PARTY's own business, not the queried company's - e.g. "Rival AMD posted...". A bare
# "competitors" with no name ("faces stiff competition from rivals") stays ambiguous and is handled by
# the INDUSTRY-style fallback below, not this list.
COMPETITOR_CUES = ("rival", "rivals", "competitor", "competitors", "competing firm", "cross-town rival")

# Transaction/relationship language. Since every document in this pipeline was fetched FOR the queried
# entity (research/providers.py's NewsProvider queries by company name), a bare "customers"/"suppliers"
# mention with no OTHER company named is, by strong prior, about THIS company's own customers/suppliers
# - "Several customers are reportedly pulling back on orders" is demand news about the queried company,
# not an independent story about the customers' own business. KNOWN FALSE POSITIVE: an article that is
# actually about a different company's customers (rare in a single-entity-queried feed, but possible if
# the source document itself covers multiple companies) will be misattributed as SELF.
COUNTERPARTY_CUES = ("customer", "customers", "client", "clients", "supplier", "suppliers", "vendor",
                      "vendors", "partner", "partners", "distributor", "distributors")

# A subsidiary/unit/division's news IS the parent company's business - explicitly attributable.
SUBSIDIARY_CUES = ("subsidiary", "subsidiaries", "business unit", "division", "unit of")

# Broad, no-single-company language. Excluded UNLESS the queried entity is also named in the same
# clause (see resolve_entity_role) - "NVIDIA warned that new export curbs could hurt the broader
# industry" is still NVIDIA news despite the industry-wide framing, because NVIDIA is named. Kept
# domain-general on purpose (no hardcoded sector words like "chipmakers") so this module isn't
# implicitly tied to any one industry.
INDUSTRY_CUES = ("industry-wide", "industrywide", "sector-wide", "sectorwide", "across the industry",
                  "across the sector", "the broader industry", "the whole industry", "the entire industry",
                  "industry peers", "sector peers", "peers across the industry")

_NAMED_PARTY_RE = re.compile(
    r"\b(?i:" + "|".join(re.escape(c) for c in COMPETITOR_CUES) + r")\s+"
    r"((?:[A-Z][A-Za-z0-9&.\-]*\s*){1,3})"
)


def _has_cue(lower_clause: str, cues: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(c)}\b", lower_clause) for c in cues)


def _entity_named(clause: str, entity: str, aliases: tuple[str, ...]) -> bool:
    """Whether the queried entity (by ticker or any supplied alias, e.g. its
    registered company name) is explicitly named in this clause. This is a
    literal-string check, not coreference - a clause that only ever says
    "the company"/"it" will not match even when it clearly means the
    queried entity. See the module docstring's disclosed limitation."""
    lower = clause.lower()
    for name in (entity, *aliases):
        if name and re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return True
    return False


def _extract_named_party(clause: str) -> str | None:
    """Best-effort: the capitalized word sequence immediately following a
    competitor cue - "Rival AMD posted..." -> "AMD". Not real NER; returns
    None rather than a low-confidence guess if nothing capitalized follows."""
    match = _NAMED_PARTY_RE.search(clause)
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


@dataclass(frozen=True)
class EventArguments:
    """subject -> action -> object -> affected-entity, approximated:
    subject_role/subject_name answer "who", object_area (the caller
    supplies affected_area, already computed by extraction.py) answers
    "what was acted upon", attributable answers "does this belong to the
    queried entity's own state at all". `action` is deliberately NOT
    re-extracted here - event_vector.py's existing sentiment/magnitude
    reading of the clause already answers "what happened", and
    duplicating that would risk two disagreeing readings of the same
    clause; this module only adds the subject/attribution axis that was
    missing."""
    subject_role: str    # "SELF" | "COUNTERPARTY" | "SUBSIDIARY" | "COMPETITOR" | "INDUSTRY"
    subject_name: str | None   # best-effort named party for COMPETITOR, else None
    attributable: bool   # whether this clause should count toward the queried entity's own state


def resolve_entity_role(clause: str, entity: str, aliases: tuple[str, ...] = ()) -> EventArguments:
    """Precedence, each checked in order (first match wins) - see the cue
    tuples above for what triggers each branch:

    1. COMPETITOR cue + a named party that isn't the queried entity itself
       -> COMPETITOR, not attributable (this clause is about THAT party).
    2. INDUSTRY cue, entity NOT named in the same clause -> INDUSTRY, not
       attributable.
    3. INDUSTRY cue, entity IS named ("NVIDIA warned new export curbs
       could hurt the broader industry") -> SELF, attributable - the
       entity is explicitly called out despite the industry framing.
    4. COUNTERPARTY cue -> COUNTERPARTY, attributable (see that cue
       tuple's own docstring for why this defaults to attributable).
    5. SUBSIDIARY cue -> SUBSIDIARY, attributable.
    6. Nothing above -> SELF, attributable (the ordinary case - most
       clauses in a single-entity-queried feed are already about that
       entity with no third-party cue present at all)."""
    lower = clause.lower()
    entity_named = _entity_named(clause, entity, aliases)

    if _has_cue(lower, COMPETITOR_CUES):
        named_party = _extract_named_party(clause)
        if named_party and named_party.lower() not in {entity.lower(), *[a.lower() for a in aliases if a]}:
            return EventArguments(subject_role="COMPETITOR", subject_name=named_party, attributable=False)

    if _has_cue(lower, INDUSTRY_CUES) and not entity_named:
        return EventArguments(subject_role="INDUSTRY", subject_name=None, attributable=False)

    if _has_cue(lower, COUNTERPARTY_CUES):
        return EventArguments(subject_role="COUNTERPARTY", subject_name=None, attributable=True)

    if _has_cue(lower, SUBSIDIARY_CUES):
        return EventArguments(subject_role="SUBSIDIARY", subject_name=None, attributable=True)

    return EventArguments(subject_role="SELF", subject_name=None, attributable=True)
