"""Data structures for the methodology-ingestion layer (stage 6). A
RawMethodologySource is the input - a short, ALREADY-paraphrased-by-us
description of a publicly documented trading methodology, never verbatim
copyrighted text (see seed_corpus.py's module docstring for the copyright
discipline this project follows). An ExtractedConceptClaim is what an
Extractor (extractor.py) produces from it: a candidate mapping onto the
canonical ontology (concepts/ontology.py), with a short rationale kept for
audit purposes only.

TRADER CLAIMS ARE HYPOTHESIS SOURCES, NEVER EVIDENCE OF PROFITABILITY -
this is enforced structurally, not just by convention: nothing in this
package has a field for "win rate", "returns claimed", or any other
performance figure a methodology's source might state. Only a concept
mapping (which canonical concept this methodology's description invokes)
is captured. Whether that concept actually predicts anything is decided
ENTIRELY by learn/hypothesis_testing.py against real, out-of-sample
episodic_events - never by how the methodology was described or how many
methodologies happen to agree.
"""
from __future__ import annotations

from dataclasses import dataclass

from market_agent.concepts.ontology import TradingConcept


@dataclass
class RawMethodologySource:
    name: str            # the methodology/system's own public name, e.g. "Darvas Box"
    practitioner: str     # who publicly documented it
    source_type: str       # 'book' | 'published_research' | 'publicly_documented_system' | 'interview'
    raw_text: str            # a short, paraphrased-by-us description (NOT verbatim source text)


@dataclass
class ExtractedConceptClaim:
    concept: TradingConcept
    rationale: str        # short, paraphrased justification - audit trail only, never trusted as fact
